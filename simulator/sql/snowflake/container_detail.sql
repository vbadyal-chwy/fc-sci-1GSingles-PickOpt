-- Pickachu Container Detail Query
with warehouse_config as (
    select wh_id, time_zone
    from edldb_dev.fulfillment_analytics_sandbox.sandbox_fulfillment_warehouses
    where wh_id = %s        -- wh_id
)

,autobatch_release as (
    select distinct
        tcpd.wh_id,      
        tcpd.container_id,
        convert_timezone('UTC', wc.time_zone, tcp.created_date) as autobatched_date_local,
        tcp.batch_id,
        convert_timezone('UTC', wc.time_zone, tcp.released_date) as released_date_local
    from edldb.aad.t_container_print_detail tcpd
    join warehouse_config wc on wc.wh_id = tcpd.wh_id
    left join edldb.aad.t_container_print tcp 
        on tcpd.wh_id = tcp.wh_id 
        and tcpd.batch_id = tcp.batch_id
    where tcpd.wh_id = %s
    and convert_timezone('UTC', wc.time_zone, tcp.released_date)::date between %s::date - 4 and %s::date + 2
)

,eligible_containers as (
    -- Containers charged during the target period
    select distinct
        pc.wh_id,
        pc.container_id
    from edldb.aad.t_pick_container pc
    join warehouse_config wc on wc.wh_id = pc.wh_id
    where pc.wh_id = %s        -- wh_id
    and convert_timezone('UTC', wc.time_zone, pc.arrive_date) between %s and %s        -- start_time and end_time
    and pc.status = 'SHIPPED'
    and pc.profile_name = 'All'
    and pc.process_path = 'Multis'
    
    union
    
    -- Containers charged before but batched after the start time
    select distinct
        pc.wh_id,
        pc.container_id
    from edldb.aad.t_pick_container pc
    join warehouse_config wc on wc.wh_id = pc.wh_id
    join autobatch_release ar 
        on ar.container_id = pc.container_id 
        and ar.wh_id = pc.wh_id
    where pc.wh_id = %s        -- wh_id
    and convert_timezone('UTC', wc.time_zone, pc.arrive_date) < %s        -- start_time
    and ar.autobatched_date_local >= %s        -- start_time
    and pc.status = 'SHIPPED'
    and pc.profile_name = 'All'
    and pc.process_path = 'Multis'
)

-- Final result with pick details
select distinct
    ec.wh_id,
    ec.container_id,
    tpd.pick_id,
    tpd.item_number,
    tpd.planned_quantity,
    tpd.pick_location,
    coalesce(fwd.status, 'N/A') as location_status,
    loc.aisle_sequence,
    loc.aisle_name,
    loc.picking_flow_as_int,
    null as optimized_pick_location
from eligible_containers ec
join edldb.aad.t_pick_detail tpd 
    on ec.wh_id = tpd.wh_id 
    and ec.container_id = tpd.container_id 
left join edldb.aad.t_location loc 
    on ec.wh_id = loc.wh_id 
    and tpd.pick_location = loc.location_id 
left join edldb.aad.t_fwd_pick fwd 
    on ec.wh_id = fwd.wh_id 
    and tpd.pick_location = fwd.location_id
left join edldb.aad.t_zone_loca zl 
    on ec.wh_id = zl.wh_id 
    and tpd.pick_location = zl.location_id
where zl.zone = 'GROUND'
order by ec.container_id, tpd.item_number;
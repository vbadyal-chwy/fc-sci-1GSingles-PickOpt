-- Pickachu Multis Containers Query
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
    where tcpd.wh_id = %s        -- wh_id
    and convert_timezone('UTC', wc.time_zone, tcp.released_date)::date between %s::date - 4 and %s::date + 2        -- start_time and end_time
)

,containers_charged_during_period as (
    select distinct
        pc.wh_id,
        pc.container_id,
        pc.priority,
        pc.che_route,
        convert_timezone('UTC', wc.time_zone, pc.arrive_date) as arrive_datetime_local,
        convert_timezone('UTC', wc.time_zone, pc.original_promised_pull_datetime) as original_promised_pull_datetime_local
    from edldb.aad.t_pick_container pc
    join warehouse_config wc on wc.wh_id = pc.wh_id
    where pc.wh_id = %s        -- wh_id
    and convert_timezone('UTC', wc.time_zone, pc.arrive_date) between %s and %s        -- start_time and end_time
    and pc.status = 'SHIPPED'
    and pc.profile_name = 'Singles'
    and pc.process_path = 'Singles'
)

,containers_charged_before_batched_after as (
    select distinct
        pc.wh_id,
        pc.container_id,
        pc.priority,
        pc.che_route,
        convert_timezone('UTC', wc.time_zone, pc.arrive_date) as arrive_datetime_local,
        convert_timezone('UTC', wc.time_zone, pc.original_promised_pull_datetime) as original_promised_pull_datetime_local
    from edldb.aad.t_pick_container pc
    join warehouse_config wc on wc.wh_id = pc.wh_id
    join autobatch_release ar on ar.container_id = pc.container_id and ar.wh_id = pc.wh_id
    where pc.wh_id = %s        -- wh_id
    and convert_timezone('UTC', wc.time_zone, pc.arrive_date) < %s        -- start_time
    and ar.autobatched_date_local >= %s        -- start_time
    and pc.status = 'SHIPPED'
    and pc.profile_name = 'Singles'
    and pc.process_path = 'Singles'
)

,combined_containers as (
    select 
        wh_id,
        container_id,
        priority,
        che_route,
        arrive_datetime_local,
        original_promised_pull_datetime_local
    from containers_charged_during_period
    
    union all
    
    select 
        wh_id,
        container_id,
        priority,
        che_route,
        arrive_datetime_local,
        original_promised_pull_datetime_local
    from containers_charged_before_batched_after
)

-- Final result
select 
    wh_id,
    container_id,
    priority,
    che_route,
    arrive_datetime_local as arrive_datetime,
    original_promised_pull_datetime_local as original_promised_pull_datetime,
    0 as released_flag,
    null as tour_id,
    null as release_datetime
from combined_containers
order by arrive_datetime, container_id;
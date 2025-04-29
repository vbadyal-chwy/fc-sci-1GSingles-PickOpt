-- Fetch containers AutoBatch date
with AutoBatch_release as (
select Distinct
         tcpd.wh_id,      
         tcpd.container_id,
         tcp.created_date as autobatched_date
from aad.t_container_print_detail tcpd
left join aad.t_container_print tcp on tcpd.wh_id = tcp.wh_id and tcpd.batch_id = tcp.batch_id
where 
tcpd.wh_id = '{fc}'
and tcp.released_date::date between '{start_time}'::date - 4 and '{start_time}'::date + 2 -- cast a wider net for dates
)
-- Container data for AB simulation arrival 
, final_set as (
(
select distinct
         pc.wh_id,
         pc.container_id,
         pc.priority,
         pc.arrive_date AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York' as arrive_datetime,
         CAST((LEFT(CAST(pc.promised_date as VARCHAR),11)||LEFT(CAST(pc.cutoff_time as VARCHAR),2)||':'||RIGHT(CAST(pc.cutoff_time as VARCHAR),2)||':00') as timestamp) 
                AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York' as cut_datetime,
         pc.original_promised_pull_datetime as pull_datetime,
         pd.item_number, 
         pd.planned_quantity as pick_quantity, 
         pd.pick_location as wms_pick_location,
         right(loc.print_zone,2)::numeric(1,0) as print_zone,
         loc.aisle_sequence,
         loc.picking_flow_as_int,
         0 as released_flag
from aad.t_pick_container as pc
left join aad.t_pick_detail as pd
on pd.wh_id = pc.wh_id and pc.container_id = pd.container_id
left Join aad.t_location as loc
on loc.wh_id = pc.wh_id and loc.location_id = pd.pick_location
left Join aad.t_zone_loca as z
on z.wh_id = pc.wh_id and z.location_id = pd.pick_location
left join aad.t_item_uom as uom 
on uom.wh_id = pc.wh_id and uom.item_number = pd.item_number
where pd.wh_id = '{fc}'
and pc.arrive_date between '{start_time}' and '{end_time}'
and pc.status in ('SHIPPED')
and distinct_items > 1
and z.zone = 'GROUND'
order by 2,3,4,1,8
)
union 
-- Containers charged but not batched before selected simulation start time
(
select distinct
         pc.wh_id,
         pc.container_id,
         pc.priority,
         pc.arrive_date AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York' as arrive_datetime,
         CAST((LEFT(CAST(pc.promised_date as VARCHAR),11)||LEFT(CAST(pc.cutoff_time as VARCHAR),2)||':'||RIGHT(CAST(pc.cutoff_time as VARCHAR),2)||':00') as timestamp) 
                AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York' as cut_datetime,
         pc.original_promised_pull_datetime as pull_datetime,
         pd.item_number, 
         pd.planned_quantity as pick_quantity, 
         pd.pick_location as wms_pick_location,
         right(loc.print_zone,2)::numeric(1,0) as print_zone,
         loc.aisle_sequence,
         loc.picking_flow_as_int,
         0 as released_flag
from aad.t_pick_container as pc
left join aad.t_pick_detail as pd
on pd.wh_id = pc.wh_id and pc.container_id = pd.container_id
left Join aad.t_location as loc
on loc.wh_id = pc.wh_id and loc.location_id = pd.pick_location
left Join aad.t_zone_loca as z
on z.wh_id = pc.wh_id and z.location_id = pd.pick_location
left join aad.t_item_uom as uom 
on uom.wh_id = pc.wh_id and uom.item_number = pd.item_number
left join AutoBatch_release as ar
on ar.container_id = pc.container_id
where pd.wh_id = '{fc}'
and pc.arrive_date < '{start_time}'
and ar.autobatched_date > '{start_time}'
and pc.status in ('SHIPPED')
and distinct_items > 1
and z.zone = 'GROUND'
order by 2,3,4,1,8
)
) 
select * from final_set order by 2,5

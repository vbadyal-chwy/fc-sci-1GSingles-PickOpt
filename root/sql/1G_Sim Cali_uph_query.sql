with preQ as (
select Distinct
         tcpd.wh_id,
         tcpd.batch_id,    
         tcpd.container_id,
         tcp.created_date as autobatched_date,
         tl.location_id,
         tl.item_number,
         tl.tran_qty,
         tl.employee_id,
         tl.start_tran_date_time,
         tl.end_tran_date_time,
         (timestampdiff(minute,tl.start_tran_date_time,tl.end_tran_date_time)/60.0)::numeric(10,2) as pick_time

from aad.t_container_print_detail tcpd
left join aad.t_container_print tcp on tcpd.wh_id = tcp.wh_id and tcpd.batch_id = tcp.batch_id
left join aad.t_pick_container pc on tcpd.wh_id = pc.wh_id and tcpd.container_id = pc.container_id
left join aad.t_tran_log tl on tcpd.wh_id = tl.wh_id and tcpd.container_id = tl.hu_id_2
left join aad.t_location loc on loc.wh_id = tl.wh_id and loc.location_id = tl.location_id
left Join aad.t_zone_loca as z on z.wh_id = tl.wh_id and z.location_id = tl.location_id and z.zone = 'GROUND'
where tcpd.wh_id = 'AVP1'
and tl.tran_type in ('301','381')
and z.zone = 'GROUND'
and tcpd.batch_id not like '%-PEX' 
and tcpd.batch_id not like 'VN%'
and tcpd.batch_id not like 'S%'
--and ((pc.arrive_date between '2025-01-13 06:00:00' and '2025-01-20 05:59:59')
--or (pc.arrive_date < '2025-01-13 06:00:00' and tcp.released_date > '2025-01-13 06:00:00'))
and tcp.created_date between '2025-01-27 6:00:00' and '2025-01-28 05:59:59'          -- adjust dates
--and pc.distinct_items > 1
)
select 
        wh_id
       ,count( distinct container_id) as count_container_id
       ,sum(tran_qty) as sum_tran_qty
       ,sum(pick_time) as total_labour_hours
       ,sum(tran_qty)/sum(pick_time) as uph 
from preQ
group by 1


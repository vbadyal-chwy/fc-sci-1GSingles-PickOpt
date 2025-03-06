select loc.wh_id,
       loc.location_id,
       si.item_number,
       si.actual_qty,
       loc.type,
       loc.print_zone,
       loc.aisle_sequence, 
       loc.picking_flow_as_int
from aad.t_location loc
left join aad.t_zone_loca z on z.wh_id = loc.wh_id and z.location_id = loc.location_id
left join aad.t_stored_item si on loc.wh_id = si.wh_id and loc.location_id = si.location_id
where loc.wh_id = '{fc}'
and z.zone = 'GROUND'
and loc.type in ('I', 'P', 'M')
and si.item_number is not null
order by 2,3

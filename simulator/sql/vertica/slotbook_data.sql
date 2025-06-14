with inv_bin_data as (
select 
        ib.fulfillment_center_code as wh_id,
        ib.inventory_snapshot_date,
        ib.product_part_number as item_number,
        ib.bin_location_id as location_id,
        loc.type,
        loc.aisle_sequence,
        loc.picking_flow_as_int,
        ib.bin_location_actual_quantity,
        ib.bin_location_unavailable_quantity,
        ib.bin_location_status,
        ib.bin_location_fifo_date
from chewybi.inventory_bin_location_snapshot ib
left join aad.t_location loc on loc.wh_id = ib.fulfillment_center_code and loc.location_id = ib.bin_location_id
left join aad.t_zone_loca zl on zl.wh_id = loc.wh_id and zl.location_id = loc.location_id
where ib.fulfillment_center_code in ('{fc}')
and ib.inventory_snapshot_date::date between '{start_date}' and '{end_date}'
and loc.picking_flow_as_int <> 0
and zl.zone = 'GROUND'
and loc.type in ('I', 'P', 'M')
and ib.product_part_number is not null
order by 1,2
) 

select wh_id,
       inventory_snapshot_date,
       item_number,
       location_id,
       aisle_sequence,
       picking_flow_as_int,
       bin_location_actual_quantity as actual_qty,
       type
from inv_bin_data
order by 1,2
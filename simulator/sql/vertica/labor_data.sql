-- Historical Labor Data for Simulation
With labor_classes as (
select
        la.labor_department_id,
        ld.labor_department,
        lf.labor_area_id,
        la.labor_area,
        lf.id as labor_function_id,
        lf.labor_function,
        lf.is_direct
from labormgmt.labor_functions lf
left join labormgmt.labor_areas la on lf.labor_area_id = la.id
left join labormgmt.labor_departments ld on la.labor_department_id = ld.id
where 
la.is_active = 'true' 
and ld.is_deleted = 'false'
)

,labor_units as (
select
        lu.warehouse,
        (lu.time_segment  at timezone 'UTC' at timezone w.time_zone) as time_segment,
        lu.labor_function_id,
        lu.employee_id,
        sum(lu.units) as units,
        sum(lu.transactions) as transactions

from labormgmt.labor_unit_segments lu
left join sandbox_fulfillment.warehouses w on lu.warehouse = w.wh_id
-- where lu.warehouse in ('AVP1')  ---- change/Add FC
where lu.warehouse = %s    --wh_id
and lu.is_exception = 'false' 
and lu.time_segment >= current_date - interval '12 weeks' 
and w.is_active = 'Y' and w.type = 'FC'
group by 1,2,3,4
)

,labor_time as (
select
        lt.warehouse,
        (lt.time_segment at timezone 'UTC' at timezone w.time_zone) as time_segment,
        lt.labor_function_id,
        lt.employee_id,
        sum(lt.time_in_seconds) as time_in_seconds

from labormgmt.labor_time_segments lt
left join sandbox_fulfillment.warehouses w on lt.warehouse = w.wh_id
-- where lt.warehouse in ('AVP1') ---- change/Add FC
where lt.warehouse = %s    --wh_id
and lt.is_exception = 'false' 
and lt.time_segment >= current_date - interval '12 weeks' 
and w.is_active = 'Y' and w.type = 'FC'
group by 1,2,3,4
)

-------------------------------------------------ABOVE GRABS EACH PART OF CLMS DATA----------------------------------------------------------------
,full_data as (
select 
        isnull(t.warehouse,u.warehouse) as warehouse,
        isnull(t.time_segment,u.time_segment) as time_segment,
        isnull(t.employee_id,u.employee_id) as employee_id,
        isnull(t.labor_function_id,u.labor_function_id) as labor_function_id,
        isnull(t.time_in_seconds,0) as time_in_seconds,
        isnull(u.units,0) as units,
        isnull(u.transactions,0) as transactions
from labor_time t
full outer join labor_units u on t.warehouse = u.warehouse 
and t.time_segment = u.time_segment 
and t.labor_function_id = u.labor_function_id 
and t.employee_id = u.employee_id
)
,clms_data as(
select 
        f.warehouse,
        f.time_segment,
        f.employee_id,
        f.labor_function_id,
        lc.labor_function,
        lc.labor_area,
        lc.labor_department,
        lc.is_direct,
        f.time_in_seconds,
        f.units as units,
        f.transactions as transactions
from full_data f
left join labor_classes lc on lc.labor_function_id = f.labor_function_id
where labor_area in ('Outbound - Multis')
)
, final_data as(
select distinct 
        warehouse,
        time_segment::date as time_segment,
        hour(time_segment) as hour_time_segment,
        minute(time_segment) as minute_time_segment,
        labor_area,
        labor_function,
        count(distinct employee_id) as count_employees,
        sum(units) as units,
        (sum(time_in_seconds)/3600.0)::numeric(10,2) as CLMS_Time_hours

from clms_data
where warehouse = %s   --wh_id
and time_segment between %s AND %s --start_date and end_date
and labor_function in ('Multis Ground Picking')
group by 1,2,3,4,5,6
order by 1,2,3,4
)

select 
        warehouse as wh_id,
        time_segment as date,
        hour_time_segment as hour,
        minute_time_segment as minutes,
        count_employees
        
from final_data order by 1,2,3,4


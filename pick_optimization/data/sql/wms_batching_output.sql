--Batch Data 
With batch_data as (
select 
        tcp.wh_id,
        tcp.created_date::date as wms_batch_created_date,                     -- Time stamp as batches are created in WMS (confirmed by Del)
        hour(tcp.created_date) as batch_create_hour,
        hour(tcp.created_date) * 3600.00 as batch_create_seconds,
        tcp.batch_id,
        dense_rank() over(partition by tcp.created_date::date  order by tcp.created_date::date ,hour(tcp.created_date),tcp.batch_id) as batch_id_1,
        count (distinct tl.hu_id_2) as CountcontainerID
from aad.t_container_print tcp
left join aad.t_tran_log tl on tcp.wh_id = tl.wh_id and tcp.batch_Id = tl.control_number_2
left join aad.t_location loc on loc.wh_id = tl.wh_id and loc.location_id = tl.location_id
left Join aad.t_zone_loca as z on z.wh_id = tl.wh_id and z.location_id = tl.location_id and z.zone = 'GROUND'
where 
tcp.wh_id = 'AVP1'                                                      -- adjust FC 
and tl.tran_type in ('301','381')
and z.zone = 'GROUND'
and tcp.created_date::date between '2025-03-01' and '2025-03-15'       -- adjust dates and use wider date range
group by 1,2,3,4,5

)


---------------------------
, PreQ as (
select 
        bd.wh_id,
        bd.wms_batch_created_date,                     -- Time stamp as batches are created in WMS (confirmed by Del)
        dense_rank() over (order by bd.wms_batch_created_date) AS date_rank,
        bd.batch_id,
        bd.batch_create_hour,
        bd.batch_create_hour*3600.00 as batch_create_seconds,
        bd.batch_id_1,                   
        dense_rank() over (order by bd.wms_batch_created_date, bd.batch_create_hour, bd.batch_id) as continuous_batch_id,
        CASE WHEN TL.control_number_2 like '%DYN%-%' THEN CASE  WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 325 and 331 THEN 'OP-SZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 332 and 338 THEN 'OP-SSZEA'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 339 and 345 THEN 'OP-SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) = 0 THEN 'HAFS N/A'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) = 1 THEN 'HAFS N/A'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)                      
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 25 and 31 THEN 'SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 32 and 38 THEN 'SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'), '\d*') as INT) between 39 and 45 THEN 'SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-DYN\d'),1)
                                                       WHEN LEFT(REGEXP_SUBSTR(TL.control_number_2,'DYN\d'), 3) = 'DYN' THEN 'HAFS N/A'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'DYN\d'),1)
                                                       ELSE '*-WAIT-*'
                                                   END       
             WHEN TL.control_number_2 like '%ac%-%' THEN CASE WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'), '\d*') as INT) between 411 and 417 THEN 'AC-SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'), '\d*') as INT) between 418 and 424 THEN 'AC-SSZEA'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'),1)
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'), '\d*') as INT) between 425 and 431 THEN 'AC-SZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'),1)   
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'), '\d*') as INT) between 446 and 451 THEN 'MC-SZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-ac\d'),1)    
                                                       else '*-WAIT-*' 
                                                       end 
             WHEN (TL.control_number_2 like '%a%-%' and TL.control_number_2 not like '%ac%-%') THEN CASE WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-a\d'), '\d*') as INT) between 432 and 438 THEN 'MC-SSZEZ'|| RIGHT(REGEXP_SUBSTR(TL.control_number_2,'\d*\d-a\d'),1) 
                                                       else '*-WAIT-*'
                                                   end                                       
 
                                                                                        
            WHEN LEFT(TL.control_number_2,3) = '01-' THEN 'SSZEA'
            WHEN LEFT(TL.control_number_2,3) = '02-' THEN 'SSZEA'                                           --- Abbreviations for this area --
            WHEN LEFT(TL.control_number_2,3) = '03-' THEN 'SSZEA'                                           --  SSZEZ = Same Start Zone End Zone
            WHEN LEFT(TL.control_number_2,3) = '04-' THEN 'SSZEA'                                           --  SSZEA = Same Start Zone End Aisle  
            WHEN LEFT(TL.control_number_2,3) = '05-' THEN 'SSZEA'                                           --  SSZ = Same Start Zone
            WHEN LEFT(TL.control_number_2,3) = '17-' THEN 'RC'                                              --  SZ = Single Zone
            WHEN LEFT(TL.control_number_2,3) = '19-' THEN 'RC'
            WHEN LEFT(TL.control_number_2,4) = '80-Z' THEN 'SSZEA'
            WHEN LEFT(TL.control_number_2,3) = '94-' THEN 'SSZEA'                                          
            WHEN LEFT(TL.control_number_2,4) = 'A103' THEN 'OP-SZ'
            WHEN LEFT(TL.control_number_2,4) = 'A105' THEN 'OP-SA'
            WHEN LEFT(TL.control_number_2,3) = '105' THEN 'OP-SA'
            WHEN LEFT(TL.control_number_2,4) = 'A109' THEN 'OP-SSZEZ'
            WHEN LEFT(TL.control_number_2,4) in ('A121','G121') THEN 'All Pickable'--EX and OP
            WHEN LEFT(TL.control_number_2,4) = 'A122' THEN 'All Unpickable'
            WHEN LEFT(TL.control_number_2,4) = 'A136' THEN 'Mixed/Air Only' --CS and OP
            WHEN LEFT(TL.control_number_2,4) = 'A191' THEN 'OP-SSZEZ'
            WHEN LEFT(TL.control_number_2,4) = 'A194' THEN 'OP-SSZEA'
            WHEN LEFT(TL.control_number_2,4) = '194-' THEN 'OP-SSZEA'
            WHEN LEFT(TL.control_number_2,4) = 'A195' THEN 'OP-Same Start Zone'
            WHEN LEFT(TL.control_number_2,4) = 'A206' THEN 'Wall Picks' -- PL and EX
            WHEN LEFT(TL.control_number_2,3) = '206' THEN 'Wall Picks'
            WHEN LEFT(TL.control_number_2,4) = 'A244' THEN 'Air Singles'
            WHEN LEFT(TL.control_number_2,4) = 'A262' THEN 'RC-2 Hours Before Cut'
            WHEN LEFT(TL.control_number_2,4) = 'A272' THEN 'RC'
            WHEN LEFT(TL.control_number_2,4) = 'A273' THEN 'RC-0 Hours Before Cut'
            WHEN LEFT(TL.control_number_2,4) = 'A275' THEN 'RC'
            WHEN LEFT(TL.control_number_2,4) = 'A276' THEN 'Current Unpickable'
            WHEN LEFT(TL.control_number_2,4) = 'A277' THEN 'AC-Mixed/Air Only'
            WHEN LEFT(TL.control_number_2,4) in ('A278','G278') THEN 'AC-All Pickable'
            WHEN LEFT(TL.control_number_2,4) = 'A279' THEN 'AC-SSZEZ'
            WHEN LEFT(TL.control_number_2,4) = 'A280' THEN 'AC-SSZEA'
            WHEN LEFT(TL.control_number_2,4) = 'A281' THEN 'AC-SSZ'--same start zone 
            WHEN LEFT(TL.control_number_2,4) = 'A285' THEN 'AC-SZ'
            WHEN LEFT(TL.control_number_2,4) = 'A284' THEN 'AC-Single Aisle'
            WHEN LEFT(TL.control_number_2,4) = 'A286' THEN 'AGED'
            WHEN LEFT(TL.control_number_2,4) = 'A312' THEN 'EX-Air Singles'
            WHEN LEFT(TL.control_number_2,3) = '317' THEN 'RC-0 Hours SSZEZ'
            WHEN LEFT(TL.control_number_2,3) = '319' THEN 'RC-2 Hours SSZEZ'
            WHEN LEFT(TL.control_number_2,3) = '347' THEN 'Single Item and Location'  
            WHEN LEFT(TL.control_number_2,1) = 'C' THEN 'CSIOC'
            WHEN LEFT(TL.control_number_2,1) = 'E' THEN 'Envelope'
            WHEN TL.control_number_2 like '%M-Z%/%' THEN 'RC -SSZEA'
            WHEN TL.control_number_2 like '%M-Z%Z%' THEN 'RC -SSZEZ'
            WHEN TL.control_number_2 like 'N%-%' THEN 'HAFS N/A' || RIGHT(LEFT(TL.control_number_2,2),1)
            WHEN RIGHT(TL.control_number_2,4) = '-PEX' THEN 'Manual Print'
            WHEN TL.control_number_2 like '%RC%' THEN 'RC'
            WHEN LEFT(TL.control_number_2,1) ='S' THEN 'Singles - PK Reinduct'
            WHEN LEFT(TL.control_number_2,1) = 'U' THEN 'UOM'
            WHEN LEFT(TL.control_number_2,2) = 'VN' THEN 'VNA OnCall'
            WHEN TL.control_number_2 like 'W%' THEN 'Walk-Thru'
            WHEN TL.control_number_2 like '%WALL%' THEN 'WALL ALL PICKABLE'
            WHEN LEFT(TL.control_number_2,1) = 'Z' THEN 'SSZEA'
            WHEN LEFT(TL.control_number_2,2) = '-Z' THEN 'SSZEA'
         ELSE 'No Wave'   
        END AS batch_rule_name,                                 --obtained from FC analytics dashboar
        tl.hu_id_2 as containerID,
        tl.item_number,
        tl.tran_qty,
        tl.location_id as location_id,
        loc.picking_flow_as_int,
        loc.aisle_sequence
from batch_data bd
left join aad.t_container_print tcp on tcp.wh_id = bd.wh_id and bd.wms_batch_created_date = tcp.created_date::date and bd.batch_id = tcp.batch_id
left join aad.t_tran_log tl on tcp.wh_id = tl.wh_id and tcp.batch_Id = tl.control_number_2
left join aad.t_pick_container pc on tl.wh_id = pc.wh_id and tl.hu_id_2 = pc.container_id
left join aad.t_location loc on loc.wh_id = tl.wh_id and loc.location_id = tl.location_id
left Join aad.t_zone_loca as z on z.wh_id = tl.wh_id and z.location_id = tl.location_id and z.zone = 'GROUND'
where 
tcp.wh_id = 'AVP1'                                                      -- adjust FC 
and tl.tran_type in ('301','381')
and z.zone = 'GROUND'
and bd.batch_id not like '%-PEX' 
and bd.batch_id not like 'VN%'
and bd.batch_id not like 'S%'
        and tcp.created_date  between '2025-03-06 14:00:00' and '2025-03-06 15:00:00'        -- adjust dates
order by 1,2,6,8
) 
,PreQ_2 as (
        select wh_id,
               wms_batch_created_date as batch_date,
               batch_create_hour as batch_hour,
               continuous_batch_id as batch_id,
               count(distinct containerID) as container_count,
               count(distinct item_number) as item_count,
               sum(tran_qty) as units_picked,
               count(distinct aisle_sequence) as distinct_aisles,
               max(aisle_sequence) - min(aisle_sequence) as aisle_span
        from PreQ
        group by 1,2,3,4
)
select wh_id,
       batch_date,
       batch_hour,
       count(distinct batch_id) as tours,
       sum(container_count) as containers,
       sum(units_picked) as units,
       sum(distinct_aisles) as aisles_in,
       sum(aisle_span) as aisles_across
from PreQ_2
group by 1,2,3;

--Batch Release Log (Validation query)
with PreQ as (
        select (log_datetime AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York')::date as batch_date,
               hour(log_datetime AT TIME ZONE 'America/New_York' AT TIME ZONE 'America/New_York') as batch_hour,
               case when (rule_name like 'VC%' or rule_name like 'PC%' or rule_name like 'RC%') then 'chase' else 'non-chase' end as chase_flag,
                *
        from aad.t_auto_batching_release_log
        where wh_id = 'AVP1'
        and log_datetime between '2025-03-06 14:00:00' and '2025-03-06 15:00:00'
        and profile_name = 'All'
        and release_now = 'YES'
        and rule_name not like '%Singles%'
        --and log_datetime = '2023-11-24 03:10:30.000'
        order by log_datetime::date, hour(log_datetime), rule_priority
        
)      
       
select wh_id,
       batch_date,
       batch_hour,
       chase_flag,
       sum(batches_released) as tours, 
       avg(overall_hourly_container_target) as containers_target,
       sum(containers_released) as containers_released
from PreQ
group by 1,2,3,4
order by 2,3;
       

           


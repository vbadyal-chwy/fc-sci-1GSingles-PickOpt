
-- Query for 1G Multis Batch Cycle Time, Aisles Traveled and Other stats 
with preQ as (
select 
        tcp.wh_id,
        tcp.created_date::date as wms_batch_created_date,                     -- Time stamp as batches are created in WMS (confirmed by Del)
        tcp.batch_id,
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
        END AS batch_rule_name,                                 --obtained from FC analytics dashboard
        CASE WHEN tcp.batch_id like '%DYN%-%' THEN CASE  WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 325 and 331 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 332 and 338 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 339 and 345 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) = 0 THEN '0'
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) = 1 THEN '1'                      
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 25 and 31 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 32 and 38 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*') as INT) between 39 and 45 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-DYN\d'), '\d*')
                                                       WHEN LEFT(REGEXP_SUBSTR(tcp.batch_id,'DYN\d'), 3) = 'DYN' THEN 'DYN'
                                                       ELSE '*-WAIT-*'
                                                       END
                                                       
           WHEN tcp.batch_id like '%ac%-%' THEN CASE  WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*') as INT) between 411 and 417 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*') as INT) between 418 and 424 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*')
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*') as INT) between 425 and 431 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*')  
                                                       WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*') as INT) between 446 and 451 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-ac\d'), '\d*')      
                                                       else '*-WAIT-*'  end     
                                                     
           WHEN (tcp.batch_id like '%a%-%' and tcp.batch_id not like '%ac%-%') THEN CASE  WHEN CAST(REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-a\d'), '\d*') as INT) between 432 and 438 THEN REGEXP_SUBSTR(REGEXP_SUBSTR(tcp.batch_id,'\d*\d-a\d'), '\d*') 
                                                        else '*-WAIT-*'  end                                    
                                                               
             WHEN LEFT(tcp.batch_id,1) = 'S' THEN 'S'
             WHEN RIGHT(tcp.batch_id,4) = '-PEX' THEN '-PEX'
             WHEN LEFT(tcp.batch_id,4) = 'A103' THEN '103'
             WHEN LEFT(tcp.batch_id,4) = 'A244' THEN '244'
             WHEN LEFT(tcp.batch_id,4) = 'A312' THEN '312'
             WHEN LEFT(tcp.batch_id,4) = 'A105' THEN '105'
             WHEN LEFT(tcp.batch_id,4) = 'A195' THEN '195'
             WHEN LEFT(tcp.batch_id,4) = 'A109' THEN '109'
             WHEN LEFT(tcp.batch_id,4) = 'A136' THEN '136'
             WHEN LEFT(tcp.batch_id,4) in ('A121','G121') THEN '121'
             WHEN LEFT(tcp.batch_id,4) = 'A122' THEN '122'
             WHEN tcp.batch_id like '%M-Z%/%' THEN 'M-Z'
             WHEN tcp.batch_id like '%M-Z%Z%' THEN 'M-Z'
             WHEN LEFT(tcp.batch_id,4) = 'A275' THEN '275'
             WHEN LEFT(tcp.batch_id,4) = 'A277' THEN '277'
             WHEN LEFT(tcp.batch_id,4) = 'A272' THEN '272'
             WHEN LEFT(tcp.batch_id,4) = 'A279' THEN '279'
             WHEN LEFT(tcp.batch_id,4) = 'A280' THEN '280'
             WHEN LEFT(tcp.batch_id,4) = '80-Z' THEN '80-Z'
             WHEN LEFT(tcp.batch_id,4) = 'A273' THEN '273'
             WHEN LEFT(tcp.batch_id,4) = 'A262' THEN '262'  
             WHEN LEFT(tcp.batch_id,3) = '347' THEN '347'
             WHEN LEFT(tcp.batch_id,3) = '319' THEN '319'
             WHEN LEFT(tcp.batch_id,3) = '317' THEN '317'
             WHEN LEFT(tcp.batch_id,3) = '17-' THEN '17'
             WHEN LEFT(tcp.batch_id,3) = '19-' THEN '19'
             WHEN LEFT(tcp.batch_id,4) = 'A281' THEN '281'
             WHEN LEFT(tcp.batch_id,4) = 'A206' THEN '206'
             WHEN LEFT(tcp.batch_id,4) in ('A278','G278') THEN '278'
             WHEN LEFT(tcp.batch_id,4) = 'A276' THEN '276'
             WHEN tcp.batch_id like 'W%' THEN 'W'
             WHEN LEFT(tcp.batch_id,3) = '206' THEN '206'
             WHEN LEFT(tcp.batch_id,3) = '105' THEN '105'
             WHEN LEFT(tcp.batch_id,3) = '05-' THEN '05'
             WHEN LEFT(tcp.batch_id,3) = '04-' THEN '04'
             WHEN LEFT(tcp.batch_id,3) = '03-' THEN '03'
             WHEN LEFT(tcp.batch_id,3) = '02-' THEN '02'
             WHEN LEFT(tcp.batch_id,3) = '01-' THEN '01'
             WHEN LEFT(tcp.batch_id,4) = 'A191' THEN '191'
             WHEN LEFT(tcp.batch_id,3) = '94-' THEN '94'
             WHEN LEFT(tcp.batch_id,4) = 'A194' THEN '194'
             WHEN LEFT(tcp.batch_id,4) = '194-' THEN '194'
             WHEN LEFT(tcp.batch_id,2) = 'VN' THEN 'VN'
             WHEN LEFT(tcp.batch_id,1) = 'Z' THEN 'Z'
             WHEN LEFT(tcp.batch_id,2) = '-Z' THEN '-Z'
             WHEN LEFT(tcp.batch_id,1) = 'E' THEN 'E'
             WHEN LEFT(tcp.batch_id,1) = 'S' THEN 'S'
             WHEN LEFT(tcp.batch_id,1) = 'U' THEN 'U'
             WHEN LEFT(tcp.batch_id,1) = 'C' THEN 'C'
             WHEN LEFT(tcp.batch_id,4) = 'A285' THEN '285'
             WHEN LEFT(tcp.batch_id,4) = 'A284' THEN '284'
             WHEN LEFT(tcp.batch_id,4) = 'A286' THEN '286'
             WHEN tcp.batch_id like '%RC%' THEN 'RC'
             WHEN tcp.batch_id like '%WALL%' THEN '441' -- Wall All Pickables
             WHEN tcp.batch_id like 'N%-%' THEN 'N'
             ELSE tcp.batch_id END as batch_rule_id,
        (timestampdiff(minute, min(tl.end_tran_date_time),Max(tl.end_tran_date_time))/60.0)::numeric(10,2) as batch_cycle_time_hrs,
        (Max(loc.aisle_sequence) - Min(loc.aisle_sequence))+ 1 as aisles_traveled_in,
        count(distinct loc.aisle_sequence) as aisles_traveled_across,
        count(distinct tl.hu_id_2) as count_container_id, 
        sum(tl.tran_qty) as total_pick_qty,
        count(distinct tl.item_number) as distinct_items_picked,
        count(distinct tl.location_id) as distinct_locations_visited
from aad.t_container_print tcp
left join aad.t_tran_log tl on tcp.wh_id = tl.wh_id and tcp.batch_Id = tl.control_number_2
left join aad.t_pick_container pc on tl.wh_id = pc.wh_id and tl.hu_id_2 = pc.container_id
left join aad.t_location loc on loc.wh_id = tl.wh_id and loc.location_id = tl.location_id
left Join aad.t_zone_loca as z on z.wh_id = tl.wh_id and z.location_id = tl.location_id and z.zone = 'GROUND'
where 
tcp.wh_id = 'AVP1'                                                      -- adjust FC 
and tl.tran_type in ('301','381')
and z.zone = 'GROUND'
and batch_id not like '%-PEX' 
and batch_id not like 'VN%'
and batch_id not like 'S%'
and tcp.created_date  between '2025-02-01 6:00:00' and '2025-03-01 05:59:59'         -- adjust dates
group by 1,2,3,4
)

select 
        wh_id 
        --,wms_batch_created_date
        ,sum(count_container_id) as count_container
        ,avg(count_container_id) as avg_containers_per_batch
        ,count(batch_id) as count_batch_id
        ,sum(batch_cycle_time_hrs) as  sum_batch_cycle_time_hrs
        ,avg(batch_cycle_time_hrs)*60.0 as avg_batch_cycle_time_mins
        ,avg(aisles_traveled_across) as avg_aisles_traveled_across
        ,avg(aisles_traveled_in) as avg_aisles_traveled_in
        ,sum(total_pick_qty) as sum_total_pick_qty
        ,sum(total_pick_qty) / sum(batch_cycle_time_hrs) as UPH
from preQ
group by 1
        

           


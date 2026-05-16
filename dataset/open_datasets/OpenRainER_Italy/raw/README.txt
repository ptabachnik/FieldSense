----------------------
  OpenRainER Dataset 
----------------------

Published:      2024-02-02 (Version 1.0)
Last edited:    2026-04-16 (Version 2.0.1)

DOI:       https://zenodo.org/doi/10.5281/zenodo.10593848
License:   https://creativecommons.org/licenses/by/4.0

------

## Authors:

Elia Covi 
    Arpae SIMC, Bologna, Italy
    https://orcid.org/0000-0001-8654-0953 
    elia.c.covi@gmail.com 

Giacomo Roversi
    Ca' Foscari University, Venice, Italy 
    CNR-ISAC, Rome, Italy
    https://orcid.org/0000-0002-6560-2307
    g.roversi@isac.cnr.it

------

## Description:

OpenRainER is an open-source dataset containing two years of data (2021 and 2022) from both conventional sensors (CS), such as radars and rain gauges, and opportunistic sensors (OS), in this case Commercial Microwave Links (CML) from the Lepida ScpA (Bologna, IT) network. Following the OpenSense[1] community recommendations, the dataset is released under a Creative Commons license (CC-BY 4.0). This will promote scientific research about OS and encourage the exploitation of CML as a source for rainfall data. OpenRainER offers the opportunity to serve as benchmark dataset for CML retrieval algorithms and validation techniques over long periods and complex terrain.

Many studies (e.g. [2] and [3]) have demonstrated that CML can be effectively utilized to retrieve quantitative rain information. In Emilia-Romagna (Italy), CML data are available in near real-time since June 2020 thanks to the collaboration between the data provider Lepida ScpA, the Hydro-Meteorological and Climate Service of Emilia-Romagna Region (Arpae-SIMC), the University of Bologna and CNR-ISAC. The combined use of CML and conventional sensors is part of the EU LIFE project CLIMAXPO. An operational precipitation product based on pycomlink[4] is currently being developed under the MODMET agreement.

------
## Extent

Spatial: Longitudes [8.5, 13.21]; Latitudes [43.4, 46.0]. Emilia-Romagna, Italy. 

Temporal: [2021/01/01, 2022/12/31]. 
------

## Contents:

CML_
    COMMERCIAL MICROWAVE LINKS
    1 min frequency received and transmitted power levels
    from Lepida ScpA CML network

RADref_
    RADAR REFLECTIVITY
    Reflectivity composite of the two Emilia-Romagna Region 
    C-band weather radars with temporal frequency of 5 min

RADrain_
    RADAR RAINFALL MAPS
    15 min accumulated rain maps

RADadj_
    GAUGE ADJUSTED RADAR
    15 min accumulated rain maps, corrected with Kriging
    of the adjustment factor (RG/WR) over the gauges

AWS_
    WEATHER STATIONS
    15 min accumulated rain, temperature, wind speed and 
    direction, relative humidity (when available)

------

## Variables

CML_
   rsl: received signal level [dBm]
   tsl: transmitted signal level [dBm]
   time: UTC time [s]
   sublink_id: sublink identifier
   cml_id: commercial microwave link identifier
   length: distance between pair of antennas [m]
   site_0_lat: site 0 latitude [degrees in WGS84]
   site_0_lon: site 0 longitude [degrees in WGS84]
   site_0_elev: ground elevation above sea level at site 0 [m a.s.l.]
   site_1_lat: site 1 latitude [degrees in WGS84]
   site_1_lon: site 1 longitude [degrees in WGS84]
   site_1_elev: ground elevation above sea level at site 1 [m a.s.l.]
   frequency: sublink frequency [MHz]
   polarization: sublink polarization

RADref_
   reflectivity: Radar reflectivity factor [dBZ]
   time: UTC time [s]
   lon: Longitude [degrees WGS84]
   lat: Latitude [degrees WGS84]
   geo_dim: Geographical limits [yLL,xLL,yUR,xUR]
   mesh_dim: Grid Mesh Size [X_mesh_size, Y_mesh_size]	
   mosaic: Mosaic description codes [1 = "Gattatico, Reggio Emilia (IT)";
	                             2 = "San Pietro Capofiume, Bologna (IT)";
				     3 = "Both"]

RADrain_
   rainfall_amount: Radar Rainfall Amount [mm]
   time: UTC time [s]
   lon: Longitude [degrees WGS84]
   lat: Latitude [degrees WGS84]
   geo_dim: Geographical limits [yLL,xLL,yUR,xUR]
   mesh_dim: Grid Mesh Size [X_mesh_size, Y_mesh_size]
   mosaic: Mosaic description codes [1 = "Gattatico, Reggio Emilia (IT)";
	                             2 = "San Pietro Capofiume, Bologna (IT)";
				     3 = "Both"] 
   
RADadj_
   rainfall_amount: Adjusted Radar Rainfall Amount [mm]
   time: UTC time [s]
   lon: Longitude [degrees WGS84]
   lat: Latitude [degrees WGS84]
   geo_dim: Geographical limits [yLL,xLL,yUR,xUR]
   mesh_dim: Grid Mesh Size [X_mesh_size, Y_mesh_size]
   mosaic: Mosaic description codes [1 = "Gattatico, Reggio Emilia (IT)";
                                     2 = "San Pietro Capofiume, Bologna (IT)";
				     3 = "Both"]

AWS_
   rainfall_amount: 15min accumulated rainfall [mm]
   temperature: measured temperature [°C]
   relative_humidity: relative humidity [%]
   wind_velocity: measured wind speed [m/s]
   wind_direction: wind_direction [degree]
   time: UTC time [s]
   id: weather station identifier
   latitude: weather station latitude [degrees WGS84]
   longiude: weather station longitude [degrees WGS84]
   elevation: ground elevation at site [m a.s.l.]

------

## References:

[1] OpenSense COST Action CA20136 Opportunistic precipitation sensing network, https://opensenseaction.eu/,  2021-2025 

[2] Roversi, G., Alberoni, P. P., Fornasiero, A., and Porcù, F.: Commercial microwave links as a tool for operational rainfall monitoring in Northern Italy, Atmos. Meas. Tech., 13, 5779–5797, https://doi.org/10.5194/amt-13-5779-2020, 2020.

[3] Nebuloni, R.; Cazzaniga, G.; D’Amico, M.; Deidda, C.; De Michele, C. Comparison of CML Rainfall Data against Rain Gauges and Disdrometers in a Mountainous Environment. Sensors, 22, 3218. https://doi.org/10.3390/s22093218,  2022.

[4] Christian Chwala, Julius Polz, Max Graf, DanSereb, nblettner, keis-f, & yboose. pycomlink/pycomlink: v0.3.2 (0.3.2). Zenodo. https://doi.org/10.5281/zenodo.4810169, 2021. 





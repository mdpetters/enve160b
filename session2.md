# Particulate Matter Exposure 

\learning{
*At the end of this lab you will be*
- familiar with regional air quality sensor networks
- able to estimate particulate matter exposure 
- able to explain sources of personal exposure to particulate matter 
- able to contextualize data within regional particulate matter measurements  
}

\prompt{
The goal is to measure the particle concentration in the air you are breathing. Most or all group members will carry portable AirBeam3 particle sensors with them for a full week starting at class time on Monday. When at home or somewhere else for an extended period, the sensor can be positioned on a flat surface. You should obviously not put the sensor in a backpack or other enclosure. In your report you should estimate the personal exposure and to compare the average concentration around you during the week with that measured at a fixed site using data either from comparable low-cost sensor networks or nearby regulatory monitors from SCAQMD/CARB.
}

~~~
<img src="/assets/airbeam3.png" alt="airbeam" style="width:300px;">
~~~

The Airbeam3 sensor detects the amount of light scattered as single particles pass through a laser. In general, larger particles scatter more light than smaller ones. These sensors are sensitive enough to detect particles larger than approximately 0.3 μm. The low-resolution size distribution measured by the sensor is related to an approximate mass concentration using a built-in algorithm. The detector at the heart of these sensors is the same as that in the widely used PurpleAir sensor. 


## Motivation

Connecting air pollutant concentrations to health impacts is complicated by variability in the
actual exposure of individuals in an area or a study. What is typically used in an epidemiological
study is the average outdoor pollutant concentration measured at a fixed monitoring site and the
health outcomes of the people that live nearby. Differences in time spent in different indoor
environments, the workplace, and in traffic, as well as variability in outdoor concentrations
where individuals work and otherwise spend their time are all important, but not readily known.
One approach used to assess average exposure is to have representative groups of people carry
lightweight portable or wearable sensors over periods of several days.

## Resources

### Manuals

Airbeam3 manual [(link)](https://www.habitatmap.org/airbeam/users-guide)


### Auxiliary Data

Regional wind data [(link)](https://www.windy.com/?33.842,-117.697,8)

Historical meteorological data [(link)](https://mesowest.utah.edu/cgi-bin/droman/mesomap.cgi?lat=34.14444&lon=-117.85000&radius=25&rawsflag=290&site=CQ073&unit=0&time=LOCAL&product=&year1=2020&month1=8&day1=06&hour1=10&currTimeChecked=)

Purple air map [(link)](https://map.purpleair.com/1/mAQI/a10/p604800/cC0#9.01/33.912/-117.4613)

CARB data 1 [(link)](https://www.arb.ca.gov/aqmis2/aqdselect.php?tab=daily)

CARB data 2 [(link)](https://aqview.arb.ca.gov/continuous-monitoring-data)

CARB emissions data [(link)](https://ww2.arb.ca.gov/applications/emissions-air-basin)

CARB pollution map [(link)](https://ww3.arb.ca.gov/carbapps/pollution-map/)

SCAQMD community data [(link)](http://www.aqmd.gov/nav/about/initiatives/environmental-justice/ab617-134/ab-617-community-air-monitoring/communities)

MATES map [(link)](https://ww2.arb.ca.gov/applications/emissions-air-basin)


## Tips and Tricks

At least for some students, the time included in the downloaded AirBeam data is in a format that Excel or Sheets can’t recognize. An example is 2023-10-02T13:20:02.000.

To adjust the format so that Excel can recognize it and allow you to include it in a graph, you should use the find and replace feature (there is probably a counterpart in Sheets, but I am unfamiliar with it). Simply type CTRL-f and then click on the “Replace” tab. 

For the format shown above, you should first remove the .000 at the end by typing .001 next to “Find what” and nothing next to “Replace with”. Click Replace All. 

Next, add “T” next to “Find what” and a space next to “Replace with”. Again click Replace All.

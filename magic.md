# Lab 3. Carbon Mass Balance

\learning{
- Explain the principle of operation for aerosol measurements.
- Explain the principle of operation for chip-based gas measurements.
- Perform emission factor measurements from combustion sources using the carbon mass balance method.
- Evaluate the emission factor on a per fuel basis in the context of the peer-reviewed literature.
}


## Required Reading

Introduction to [Combustion Emissions](assets/combustion.pdf).

## Background

PM stands for particulate matter (also called particle pollution): the term for a mixture of solid particles and liquid droplets found in the air. Some particles, such as dust, dirt, soot, or smoke, are large or dark enough to be seen with the naked eye. Others are so small they can only be detected using an electron microscope. Particle pollution includes $PM_{10}$, which are inhalable particles, with diameters that are generally 10 micrometers and smaller and $PM_{2.5}$, which are fine inhalable particles, with diameters that are generally 2.5 micrometers and smaller. (Source: [EPA](https://www.epa.gov/pm-pollution/particulate-matter-pm-basics)).

~~~
<img src="/assets/pm2.5_scale_graphic-color_2.jpg">
~~~

Particulate matter is directly emitted in the form of smoke from combustion processes. This includes (wild)fires, residential and commercial cooking, and combustion engines. The size distribution and chemical composition of the formed particulate matter is complex and depends on the fuel and combustion process. A first estimate of the source strength of a process is the emission factor.

An emissions factor is a representative value that attempts to relate the quantity of a pollutant released to the atmosphere with an activity associated with the release of that pollutant. These factors are usually expressed as the weight of pollutant divided by a unit weight, volume, distance, or duration of the activity emitting the pollutant (e.g., kilograms of particulate emitted per megagram of coal burned). Such factors facilitate estimation of emissions from various sources of air pollution. In most cases, these factors are simply averages of all available data of acceptable quality, and are generally assumed to be representative of long-term averages for all facilities in the source category (i.e., a population average). (Source: [EPA](https://www.epa.gov/air-emissions-factors-and-quantification/basic-information-air-emissions-factors-and-quantification)).

In the atmosphere or indoor environment, air often dilutes by mixing. Emission factors can be estimated using the simplified carbon balance method, which assumes that all carbon in the fuel is converted to $CO_2$. In this method the change in observed property (ozone $O_3$, particle number $N$, or particle mass $PM_{2.5}$) is normalized by the change in $CO_2$. The increase in $CO_2$ serves as a measure of fuel combusted and accounts for the dilution with background air. (During combustion the vast majority of carbon is converted to $CO_2$). An example is shown below.

~~~
<img src="/assets/number_emission.png" style="width:300px;">
~~~

**Figure** (Source: [Shen et al., 2022](https://doi.org/10.1016/j.scitotenv.2021.151609)). Time series of pollutant signals. The vertical dashed line indicates the time at which a vehicle passes the sensor platform.

The figure shows a time series of particle number concentration, $NO$, and $CO_2$. The dashed horizontal line is the baseline concentration. In the plume, the particle number concentration increased from $4,500\; cm^{-3}$ (baseline) to $19,000\; cm^{-3}$ (peak plume conditions). The change is $\Delta N = N_{plume} - N_{baseline} = 19,000 - 4,500\; cm^{-3} = 14,500\; cm^{-3}$. At the same time, $CO_2$ increased from $427\; ppm$ to $470\; ppm$. The change is $\Delta CO_2 = CO_{2, plume} - CO_{2,baseline} = 470 - 427\; ppm = 43\; ppm$. Based on these changes, the emission factor can be calculated using the carbon mass balance method, which is described in detail in the required reading. 

\prompt{
The objective is to measure the emission factor of a number of gases and particulate matter from combustion sources of your choice. You will take a set of chip-based sensors bundled in the UCR Air Quality Kiosk. Expose the Kiosk to emission. Examples include smoke from candles, incense,  vehicle emissions near a tailpipe (e.g. in a parking lot or from a bus), or cooking emissions from a grill. You will carry those sensors to a suitable site. First, you will measure and record the background concentrations far from the plume. You will need to carefully consider how you define background conditions. Next, you will try to get close to the source where you can observe an increase in $CO$ and/or $CO_2$ above background. Finally, place the remaining sensors to get a measurement of the concentrations in the plume. You should familiarize yourself with the sensors on Monday and evaluate some practice targets. You may take the Kiosk home and target emissions there if you like. Discuss your preliminary results with the instructor and/or TA. Show preliminary data analysis on Wednesday and adjust the measurement protocol based on lessons learned. 
}

## Report 

Please follow the [report outline](https://docs.google.com/document/d/1PhQsRvCM1_sqDcy18_a6RG8RHiOktbe_sn7TnOJm_G4/edit?usp=sharing).

Don't forget to print and paste the [grading rubric](https://docs.google.com/document/d/1ocOcguu2LU4kSziEKwV7fkrkh7znVgPouguTiv_PYms/edit?usp=sharing) at the end of your report. Not providing the grading rubric will lead to point deductions. Use the grading rubric as a checklist for how to prepare a proper report.

## Methods

The UCR Air Quality Kiosk contains chip-based sensors for $PM$, $CO_2$, $CO$, $HCHO$ (formaldehyde), $SO_2$, $H_2S$, $NO$, and $NO_2$. Details about these sensors are provided in the required reading. During your experiment, not all sensors may show measurable signal. Make sure that your are measuring in regimes that do not saturated the sensor. Data on the Kiosk are stored to a USB disk and are reported as a comma separated data file. You can remove the USB stick, copy the data file, and then use computer software to load the time series and plot the data.

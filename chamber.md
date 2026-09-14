# Photochemical Cycle of $NO_2$, $NO$, and $O_3$

\learning{
- Perform photochemical reactions inside environmental chambers.
- Explain the principle of operation of $O_3$ and $NO_x$ gas-phase monitors.
- Explain the photochemical cycle of $O_3$, $NO$, and $NO_2$.
- Measure and calculate the ozone produced at the photostationary state.
}

## Required Reading

Chapter 12, Ozone Air Pollution, from [Introduction to Atmospheric Chemistry](https://acmg.seas.harvard.edu/education-introduction-atmospheric-chemistry/).



## Background

![Ozone Photochemistry](/assets/ozone_photochemistry_small.png)

Ozone is a gas composed of three atoms of oxygen ($O_3$). Ozone in the air we breathe can harm our health, especially on hot sunny days when ozone can reach unhealthy levels. People at greatest risk of harm from breathing air containing ozone include people with asthma. Elevated exposures to ozone can affect sensitive vegetation and ecosystems, including forests, parks, wildlife refuges and wilderness areas.  In particular, ozone can harm sensitive vegetation during the growing season. (Source: [EPA](https://www.epa.gov/ground-level-ozone-pollution/ground-level-ozone-basics)). Tropospheric $O_3$ is generated from to major classes of precursor molecules: volatile organic compounds (VOCs) and nitrogen oxides $NO_x$, which is the sum of $NO$ and $NO_2$. Most of the direct emissions on $NO_x$ are in the form of $NO$. Ozone levels exceeding 200 ppb are considered severe air pollution episodes. The current U.S. National Ambient Air Quality Standard for $O_3$ is an 8-hour average of 70 ppb. Understanding VOC chemistry is critical to explain the total ambient $O_3$ concentrations. Nevertheless, the photochemical cycle of $NO_2$, $NO$, and $O_3$ remains the starting point for modeling tropospheric $O_3$ production.


### Photochemical $O_3$ Production from $NO_x$

Sunlight at wavelength < 424 nm photolizes $NO_2$ into $NO$ and atomic $O$

$$
NO_2 + h\nu \rightarrow NO + O
$$

The atomic $O$ reacts with oxygen $O_2$ to form $O_3$. Reaction (2) is the only source of atmospheric $O_3$.

$$
O + O_2 + M \rightarrow O_3 + M
$$

where $M$ is a third body required to stabilize the excited product $OO_2^\star$ by collision. Finally $O_3$ reacts with $NO$ to regenerate $NO_2$

$$
O_3 + NO \rightarrow NO_2 + O_2
$$ 

Reactions (1)-(3) form the basic photochemical $NO_x$ cycle. Cycling between $NO$ and $NO_2$ takes place in the troposphere on a time scale of a minute in the daytime. There is no net production of $O_3$, but some $O_3$ is present. The steady state $O_3$ concentration is 

$$
 [O_3] = -\frac{1}{2} \left ( [NO]_0 - [O_3]_0 + \frac{j_{NO_2}}{k_3} \right ) \\
      + \frac{1}{2} \left [ \left ( [NO]_0 - [O_3]_0 + \frac{j_{NO_2}}{k_3} \right )^2 + 4 \frac{j_{NO_2}}{k_3} \left ( [NO_2]_0 + [O_3]_0 \right ) \right ]^{1/2}
$$

The ratio $j_{NO_2}/k_3$ depends on sunlight or blacklight intensity. For full blacklight intensity in the chamber used in this lab, $j_{NO_2}/k_3 \approx 10.2\; ppb$. The characteristic relaxation time to steady state is 

$$
\tau = \frac{1}{k_3 [NO]}
$$

where $k_3 = 1.9\times 10^{-14} \; cm^{3}\;  molecule^{-1} \; s^{-1}$ at $T = 298K$.

### Reaction Kinetics

The photolysis rate $j_{NO_2}$ depends on the actinic flux (intensity of sunlight or intensity of blacklights). At noon in the cloud-free atmosphere $j_{NO_2} \approx 8\times 10^{-3}\; s^{-1}$ and otherwise lower. The value for $k_3 = 1.9\times 10^{-14} \; cm^{3}\;  molecule^{-1} \; s^{-1}$ at $T = 298K$. The figure below shows a typical evolution of $NO_2$, $NO$, and $O_3$ for the experiment you are going to perform. At $t = 0$ the conditions are $[O_3]_0 = 0\; ppb$, $[NO]_0 = 60\; ppb$, and $[NO_2]_0 = 450\; ppb$. The blacklights are turned on at $t = 0$. Predictions for the photostationary state concentration and relaxation time based on Eqs. (4) and (5) are provided. After 5 min, the lights are turned off and the system restores to the initial state. 

![Simulation Results](/assets/simulation.png)

\prompt{
The goal is to measure the photostationary steady state of the $NO_2$, $NO$, and $O_3$ system. The environmental chamber will be filled with zero air that is free of VOCs and other chemicals. A mix of $[NO]_0$  $[NO_2]_0$ and $[O_3]_0$ will be added to the chamber and the system will be allowed to equilibrate. Concentrations of $NO_2$, $NO$, and $O_3$ will be monitored in real time using gas-analyzers. After the system is equilibrated, blacklights will be turned on. The blacklights provide photons $\lambda < 424\; nm$, thus initiating the photolysis of $NO_2$ and the production of $O_3$. Observe the formation of $O_3$ until the photostationary state of the system is reached. Then turn off the light and observe the reaction of $NO$ with $O_3$, returning the system back to the initial state. Repeat the cycle at least once, preferably twice to ensure consistent results.

*During the Lab*

- Make sure the $O_3$ and $NO_x$ gas analyzers are running. Make sure that the data acquisition system is turned on and recording data. Compare the computer logged data on the screen to the values reported on the analyzers.  Fill the chamber with zero air. Verify that initial concentrations are approximately zero. Add the $NO$/$NO_2$ mixture to the chamber. Record the initial $NO$, $NO_2$ and $O_3$ concentrations. Turn on the light and let the system equilibrate to the photostationary steady state. Turn off the light and observe reversion to the initial state. 
- Make a draft of the schematic drawing of the experimental setup. Carefully indicate (1) All of the instruments involved, including vacuum pumps, instruments, flow rates, flow paths, light sources (location and number), compressed air sources, and data acquisition system. (2) How the reactants are added to the chamber, (3) timings of reactants added and lights turning on and off. You should make a sketch during the experiment.
- Make a draft of the schematic drawing of the O3 analyzer and discuss with your group and/or TA. 
- Download the data from the computer for later analysis.
}

## Report 

Please follow the [report outline](https://docs.google.com/document/d/1IzfiSnkv9lwBkknXHKjJsLFzyCnfKwh2Q_X9xTLBZG4/edit?usp=sharing).

Don't forget to print and paste the [grading rubric](https://docs.google.com/document/d/13-KuUIEgavq18SuWSS4LJvFvVdt0BesU7G78qKYnscY/edit?usp=sharing) at the end of your report. Not providing the grading rubric will lead to point deductions. Use the grading rubric as a checklist for how to prepare a proper report.


## Methods

This lab uses an environmental chamber, a photometric UV absorption $O_3$ analyzer and a chemiluminescence $NO_x$ analyzer. 

~~~
<img src="/assets/chamber.jpg" alt="Chamber" style="width:180px;">
~~~

~~~
<img src="/assets/ozone.jpg" alt="POM" style="width:220px;">
~~~

~~~
<img src="/assets/nox.png" alt="airbeam" style="width:190px;">
~~~

The environmental chamber consists of a Teflon bag housed inside an enclosure that is lined with blacklights. The blacklights can be turned on/off using switches. An injection port is available to add reagents. You will receive a glass bulb that contains a nominal amount of $NO$. You will add the $NO$ with the help of the TA and/or the instructor. Ozone is measured using a gas-analyzer. A 254 nm UV light signal is passed through the sample cell where it is absorbed in proportion to the amount of ozone present. Periodically, a switching valve alternates measurement between the sample stream and a sample that has been scrubbed of ozone. $NO_x$ is measured using a second gas analyzer. The $NO_x$ instrument determines the concentration of nitric oxide ($NO$), total nitrogen oxides ($NO_x$) , the sum of $NO$ and $NO_2$) and nitrogen dioxide ($NO_2$) in a sample stream. The principle of operation is [chemiluminescence](https://en.wikipedia.org/wiki/Chemiluminescence). Chemiluminescence is the emission of light from a chemical reaction and is triggered by the reaction of $NO$ with ozone $O_3$. The amount of light produced is linear with $NO$ concentration. $NO_2$ is measured by converting $NO_2$ with to $NO$ using heated molybdenum converter chip. 

## Resources

Teledyne Product Manuals ($O_3$ analyzer and $NO_x$ analyzer) [link](https://www.teledyne-api.com/en-us/service-support/product-manuals)


Content
The demonstrator sorts different two different materials (wood and metal) into their corresponding target locations. The different modules can be placed at any of the four positions and the PLC program automatically adjusts for the change in location.

The four modules are:

Storage Magazine
Sensor
Metal storage
Wood storage
A linear drive with a pneumatic gripper transports the materials between the different stations.

The procedure follows these steps:

0) Idle, waiting for start button press.

1) Homing, do homing when drive is not homed yet (only after first power on).

2) Move gripper to a position next to the storage module. This avoids collisions of the storage slider and the gripper when
material is ejected.

3) Eject material.

4) Move gripper to storage position.

5) Close the gripper to pick up material.

6) Move to the sensor to detect material type.

7) Move to the corresponding storage box of the detected material.

8) Open the gripper to release material.



Picture of the Genesis demonstrator


First dataset contains files Genesis_StateMachineLabels.csv and Genesis_AnomalyLabels.csv:

In this dataset the drive was already homed and the stations were positioned in the following order:

Position 1: Storage Module

Position 2: Wood Storage

Position 3: Metal storage

Position 4: Sensor Module



Both data sets contain 16220 observations taken every 50ms through an OPC DA server.
They are identical with just the labels being different.
Missing values and Zero-Variance columns are already removed from the data.

The Label column in Genesis_StateMachineLabel.csv represents the internal state machine of the PLC code.

State Machine description:

0: Idle,

1: Homing,

2: AvoidStorage,

3: ActivateStorage,

4: ToStorage,

5: CloseGripper,

6: ToSensor,

7: ToBox,

8: OpenGripper



In the Genesis_AnomalyLabels.csv file the anomaly Labels are manually annotated, checked very carefully and are accurate for each data point!

Only one type of Anomaly was simulated.

Anomaly description:

0: No anomaly

1: Linear drive jammed / tilted

2: Linear drive breaks free and corrects accumulated lag error



Table with production cycles




Second dataset contains files Genesis_normal.csv, Genesis_lineardrive.csv, Genesis_pressure.csv:

This dataset contains unlabelled data that contains files with normal runs and runs with errors.

In the Genesis_normal.csv file, the Demonstrator worked as intended, without any failures or restrictions. It can be used to compare it with other files for predictive maintenance or anomaly detection.

In the Genesis_lineardrive.csv file, the lineardrive was slightly impaired over time, so that the Genesis Demonstrator does not work as intended. This file can be used for predictive maintenance or anomaly detection.

In the Genesis_pressure.csv file, the air pressure was reduced over time, so that the Genesis Demonstrator does not work as intended. This file can be used for predictive maintenance or anomaly detection.



Both datasets are not necessarily compatible with each other.

Acknowledgements
© Copyright | inIT - Institute Industrial IT

© Copyright | Ostwestfalen-Lippe University of Applied Sciences

This dataset is publicly available for anyone to use under the following terms.

von Birgelen, Alexander; Niggemann, Oliver: Anomaly Detection and Localization for Cyber-Physical Production Systems with Self-Organizing Maps. S.: 55-71, Springer Vieweg, Aug 2018.
https://www.hs-owl.de/init/veroeffentlichungen/publikationen/a/filteroff/3373/single.html

IMPROVE has received funding from the European Union's Horizon 2020 research and innovation programme under Grant Agreement No. 678867




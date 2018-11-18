RoboJam: A Mixture Density RNN for creating touchscreen performances
====================================================================

[![DOI](https://zenodo.org/badge/110691324.svg)](https://zenodo.org/badge/latestdoi/110691324)

RoboJam is a Mixture Density RNN and web app for creating and responding to musical touchscreen performances.
The RNN design here is a novel application of mixture density network (MDN) to musical touchscreen data.
This data consists of a sequence of touch interaction events in the format `[x, y, dt]`. 
This network learns to predict these events so that a user's interaction can be continued from where they leave off.
The web app runs uses Flask with a public API that can be used for interaction with touchscreen music apps running on phones or tablets.
More information is in the paper (to be added soon).

Have a look at [how RoboJam is used in a touchscreen app](https://vimeo.com/242251501).

Here's an example:

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_unconditioned_1.png)

Data Format.
------------

Touchscreen performances should be stored in numpy arrays in the following format:

  [x, y, dt]
  
Where `x` and `y` are in [0,1] and `dt` is in [0,5].

Todo:
-----

- Implement freezing model for more convenient loading in server.
- Implement restart training from checkpoint
- Include links to pre-processed data for training and validation.

Examples:
---------

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_conditioned_1.png)

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_conditioned_2.png)

![](https://github.com/cpmpercussion/robojam/raw/master/notebooks/example_unconditioned_2.png)

Installation:
-------------

    pip3 install -r ./requirements.txt

Running:
--------

    python3 serve_tiny_performance_mdrnn.py

Testing:
--------

    curl -i -k -X POST -H "Content-Type:application/json" https://127.0.0.1:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.005213, 0.711230, 0.070856, 25.524292, 0\n0.097298, 0.719251, 0.062834, 25.524292, 1\n0.126225, 0.719251, 0.057487, 25.524292, 1\n0.194616, 0.707219, 0.045455, 38.290771, 1\n0.212923, 0.704545, 0.045455, 38.290771, 1\n0.343579, 0.703209, 0.108289, 38.290771, 1\n0.495085, 0.701872, 0.070856, 38.290771, 1\n0.523921, 0.693850, 0.061497, 38.290771, 1\n0.712066, 0.711230, 0.155080, 38.290771, 1\n0.730294, 0.717914, 0.155080, 38.290771, 1\n0.896367, 0.696524, 0.041444, 38.290771, 1\n1.083786, 0.696524, 0.151070, 38.290771, 1\n1.301470, 0.684492, 0.049465, 38.290771, 1\n1.328134, 0.680481, 0.053476, 38.290771, 1\n1.504139, 0.705882, 0.136364, 38.290771, 1\n1.527875, 0.712567, 0.120321, 38.290771, 1\n1.702672, 0.675134, 0.076203, 38.290771, 1\n1.719294, 0.675134, 0.096257, 38.290771, 1\n1.901434, 0.715241, 0.145722, 38.290771, 1\n1.922717, 0.717914, 0.136364, 38.290771, 1\n2.062994, 0.684492, 0.109626, 38.290771, 1\n2.091680, 0.680481, 0.129679, 38.290771, 1\n2.231362, 0.697861, 0.207219, 38.290771, 1\n2.393213, 0.712567, 0.124332, 38.290771, 1\n2.525774, 0.632353, 0.149733, 38.290771, 1\n2.546701, 0.625668, 0.169786, 38.290771, 1\n2.686487, 0.585561, 0.360963, 38.290771, 1\n2.715316, 0.580214, 0.387701, 38.290771, 1\n2.867526, 0.490642, 0.633690, 38.290771, 1\n2.880361, 0.481283, 0.645722, 38.290771, 1\n3.054443, 0.319519, 0.689840, 38.290771, 1\n3.218741, 0.121658, 0.585561, 38.290771, 1\n3.230362, 0.102941, 0.557487, 38.290771, 1\n3.391456, 0.089572, 0.534759, 38.290771, 1"}'
    curl -i -k -X POST -H "Content-Type:application/json" https://138.197.179.234:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.005213, 0.711230, 0.070856, 25.524292, 0\n0.097298, 0.719251, 0.062834, 25.524292, 1\n0.126225, 0.719251, 0.057487, 25.524292, 1\n0.194616, 0.707219, 0.045455, 38.290771, 1\n0.212923, 0.704545, 0.045455, 38.290771, 1\n0.343579, 0.703209, 0.108289, 38.290771, 1\n0.495085, 0.701872, 0.070856, 38.290771, 1\n0.523921, 0.693850, 0.061497, 38.290771, 1\n0.712066, 0.711230, 0.155080, 38.290771, 1\n0.730294, 0.717914, 0.155080, 38.290771, 1\n0.896367, 0.696524, 0.041444, 38.290771, 1\n1.083786, 0.696524, 0.151070, 38.290771, 1\n1.301470, 0.684492, 0.049465, 38.290771, 1\n1.328134, 0.680481, 0.053476, 38.290771, 1\n1.504139, 0.705882, 0.136364, 38.290771, 1\n1.527875, 0.712567, 0.120321, 38.290771, 1\n1.702672, 0.675134, 0.076203, 38.290771, 1\n1.719294, 0.675134, 0.096257, 38.290771, 1\n1.901434, 0.715241, 0.145722, 38.290771, 1\n1.922717, 0.717914, 0.136364, 38.290771, 1\n2.062994, 0.684492, 0.109626, 38.290771, 1\n2.091680, 0.680481, 0.129679, 38.290771, 1\n2.231362, 0.697861, 0.207219, 38.290771, 1\n2.393213, 0.712567, 0.124332, 38.290771, 1\n2.525774, 0.632353, 0.149733, 38.290771, 1\n2.546701, 0.625668, 0.169786, 38.290771, 1\n2.686487, 0.585561, 0.360963, 38.290771, 1\n2.715316, 0.580214, 0.387701, 38.290771, 1\n2.867526, 0.490642, 0.633690, 38.290771, 1\n2.880361, 0.481283, 0.645722, 38.290771, 1\n3.054443, 0.319519, 0.689840, 38.290771, 1\n3.218741, 0.121658, 0.585561, 38.290771, 1\n3.230362, 0.102941, 0.557487, 38.290771, 1\n3.391456, 0.089572, 0.534759, 38.290771, 1"}'
    curl -i -k -X POST -H "Content-Type:application/json" https://138.197.179.234:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.002468, 0.106414, 0.122449, 20.000000, 0\n0.020841, 0.106414, 0.125364, 20.000000, 1\n0.043218, 0.107872, 0.137026, 20.000000, 1\n0.065484, 0.107872, 0.176385, 20.000000, 1\n0.090776, 0.107872, 0.231778, 20.000000, 1\n0.110590, 0.109329, 0.301749, 20.000000, 1\n0.133338, 0.115160, 0.357143, 20.000000, 1\n0.155677, 0.125364, 0.412536, 20.000000, 1\n0.178238, 0.134111, 0.432945, 20.000000, 1\n0.516467, 0.275510, 0.180758, 20.000000, 0\n0.542726, 0.274052, 0.205539, 20.000000, 1\n0.560772, 0.274052, 0.249271, 20.000000, 1\n0.583259, 0.282799, 0.316327, 20.000000, 1\n0.605750, 0.295918, 0.376093, 20.000000, 1\n0.628259, 0.309038, 0.415452, 20.000000, 1\n0.653835, 0.316327, 0.432945, 20.000000, 1\n0.673523, 0.325073, 0.440233, 20.000000, 1\n1.000294, 0.590379, 0.179300, 20.000000, 0\n1.022137, 0.593294, 0.183673, 20.000000, 1\n1.044706, 0.594752, 0.208455, 20.000000, 1\n1.067020, 0.606414, 0.279883, 20.000000, 1\n1.091137, 0.626822, 0.355685, 20.000000, 1\n1.111968, 0.647230, 0.425656, 20.000000, 1\n1.134535, 0.655977, 0.462099, 20.000000, 1\n1.156987, 0.657434, 0.485423, 20.000000, 1\n1.619212, 0.857143, 0.263848, 20.000000, 0\n1.642492, 0.854227, 0.281341, 20.000000, 1\n1.663123, 0.851312, 0.320700, 20.000000, 1\n1.685776, 0.846939, 0.413994, 20.000000, 1\n1.708192, 0.846939, 0.510204, 20.000000, 1\n1.730717, 0.858601, 0.591837, 20.000000, 1\n1.753953, 0.868805, 0.632653, 20.000000, 1\n1.775862, 0.876093, 0.660350, 20.000000, 1\n4.376275, 0.542274, 0.860058, 20.000000, 0\n4.419554, 0.543732, 0.860058, 20.000000, 1"}'
    curl -i -k -X POST -H "Content-Type:application/json" https://0.0.0.0:5000/api/predict -d '{"perf":"time,x,y,z,moving\n0.002468, 0.106414, 0.122449, 20.000000, 0\n0.020841, 0.106414, 0.125364, 20.000000, 1\n0.043218, 0.107872, 0.137026, 20.000000, 1\n0.065484, 0.107872, 0.176385, 20.000000, 1\n0.090776, 0.107872, 0.231778, 20.000000, 1\n0.110590, 0.109329, 0.301749, 20.000000, 1\n0.133338, 0.115160, 0.357143, 20.000000, 1\n0.155677, 0.125364, 0.412536, 20.000000, 1\n0.178238, 0.134111, 0.432945, 20.000000, 1\n0.516467, 0.275510, 0.180758, 20.000000, 0\n0.542726, 0.274052, 0.205539, 20.000000, 1\n0.560772, 0.274052, 0.249271, 20.000000, 1\n0.583259, 0.282799, 0.316327, 20.000000, 1\n0.605750, 0.295918, 0.376093, 20.000000, 1\n0.628259, 0.309038, 0.415452, 20.000000, 1\n0.653835, 0.316327, 0.432945, 20.000000, 1\n0.673523, 0.325073, 0.440233, 20.000000, 1\n1.000294, 0.590379, 0.179300, 20.000000, 0\n1.022137, 0.593294, 0.183673, 20.000000, 1\n1.044706, 0.594752, 0.208455, 20.000000, 1\n1.067020, 0.606414, 0.279883, 20.000000, 1\n1.091137, 0.626822, 0.355685, 20.000000, 1\n1.111968, 0.647230, 0.425656, 20.000000, 1\n1.134535, 0.655977, 0.462099, 20.000000, 1\n1.156987, 0.657434, 0.485423, 20.000000, 1\n1.619212, 0.857143, 0.263848, 20.000000, 0\n1.642492, 0.854227, 0.281341, 20.000000, 1\n1.663123, 0.851312, 0.320700, 20.000000, 1\n1.685776, 0.846939, 0.413994, 20.000000, 1\n1.708192, 0.846939, 0.510204, 20.000000, 1\n1.730717, 0.858601, 0.591837, 20.000000, 1\n1.753953, 0.868805, 0.632653, 20.000000, 1\n1.775862, 0.876093, 0.660350, 20.000000, 1\n4.376275, 0.542274, 0.860058, 20.000000, 0\n4.419554, 0.543732, 0.860058, 20.000000, 1"}'

import datetime

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

from ..plotter import TwoDPlotter
from ...models.measurement.linear import LinearGaussian
from ...types.track import Track
from ...types.groundtruth import GroundTruthPath
from ...types.state import State, GaussianState, ParticleState
from ...types.detection import Detection
from ...types.array import StateVectors


def test_twodplotter():
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0, 1], noise_covar=np.diag([1, 1]))

    plotter = TwoDPlotter(track_indices=[0, 1],
                          detection_indices=[0, 1],
                          gtruth_indices=[0, 1],
                          measurement_model=measurement_model)
    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)

    tracks = {Track(states=[GaussianState(state_vector=[[1], [2]],
                                          timestamp=timestamp1 + datetime.timedelta(
                                          seconds=i),
                                          covar=np.diag([0.5, 0.5])) for i in range(11)])}
    tracksB = {Track(states=[ParticleState(
        state_vector=StateVectors([[1], [2]]),
        timestamp=timestamp1 + datetime.timedelta(
            seconds=i)) for i in range(11)])}
    truths = {GroundTruthPath(states=[State(np.array([[1], [2]]),
                                            timestamp=timestamp1 +
                                            datetime.timedelta(seconds=i))
                                      for i in range(11)])}
    dets = {Detection(np.array([[1], [2]]),
                      timestamp=timestamp1+datetime.timedelta(seconds=i))
            for i in range(11)}

    metrics = []

    metricA = plotter.plot_tracks_truth_detections(tracks, truths, dets)
    metrics.append(metricA)

    metricB = plotter.plot_tracks_truth_detections(tracks, truths, dets, uncertainty=True)
    metrics.append(metricB)

    metricC = plotter.plot_tracks_truth_detections(tracksB, truths, dets, particle=True)
    metrics.append(metricC)

    metricD = plotter.plot_tracks_truth_detections(tracks=None, groundtruth_paths=truths,
                                                   detections=dets)
    metrics.append(metricD)

    metricE = plotter.plot_tracks_truth_detections(tracks=tracks, groundtruth_paths=None,
                                                   detections=dets)
    metrics.append(metricE)

    for metric in metrics:
        assert metric.title == "Track plot"
        assert metric.generator == plotter
        assert isinstance(metric.value, matplotlib.figure.Figure)
        assert metric.time_range.start_timestamp == timestamp1
        assert metric.time_range.end_timestamp == timestamp2

        plt.close(metric.value)

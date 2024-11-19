import pandas as pd
import numpy as np
import csv
import dateutil

from collections import defaultdict

from math import ceil, pi
from datetime import timedelta, datetime
from typing import Tuple
from matplotlib import pyplot as plt

from pymap3d import geodetic2enu
from stonesoup.reader import DetectionReader, GroundTruthReader
from stonesoup.base import Property
from stonesoup.types.detection import Detection, Clutter
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.functions import cart2sphere, sphere2cart
from stonesoup.types.angle import Bearing
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.plotter import AnimatedPlotterly, Plotter


METERS_in_NM = 1852

# Travis Location
lat0, lon0, alt0 = 38.25049, -121.92474, 40
default_variance = 50 # estimate of variance in m2 of state matrix elements (position and velocity)

def generate_timestamps(start_time, end_time):
    total_seconds = (end_time - start_time).total_seconds()
    return [start_time + timedelta(seconds=n) for n in range(ceil(total_seconds))]


class CSVReaderXY(DetectionReader):
    rdp_file: str = Property(doc="File with the radar data.")
    ndim_state: int = Property(default=4)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tutorial 6 Approach
        measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[default_variance, 0],
                                [0, default_variance]])
            )
        
        self.model=measurement_model

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(self.rdp_file, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['timestamp']:
                    continue

                timestamp = dateutil.parser.parse(row['timestamp'], ignoretz=True)
                # lat = float(row['latitude'])
                # lon = float(row['longitude']) 
                x = float(row['x'])*METERS_in_NM
                y = float(row['y'])*METERS_in_NM
                if row['pass_no']:
                    pass_no = int(row['pass_no'])
                else:
                    pass_no=0

                metadata = {
                    'cal': float(row['cal']),
                    'sensor': 'RDU103', 
                    'pass_no': pass_no
                    }
                
                yield timestamp, {Detection(
                    [x, y], timestamp=timestamp, 
                    metadata=metadata,
                    measurement_model=self.model)}


class CSVClutterReaderXY(DetectionReader):
    rdp_file: str = Property(doc="File with the clutter data.")
    ndim_state: int = Property(default=4)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))
    # vel_mapping: Tuple[int, int] = Property(default=(1, 3))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tutorial 6 Approach
        measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[default_variance, 0],
                                [0, default_variance]])
            )
        
        self.model=measurement_model

    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(self.rdp_file, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['timestamp']:
                    continue

                timestamp = dateutil.parser.parse(row['timestamp'], ignoretz=True)
                # lat = float(row['latitude'])
                # lon = float(row['longitude']) 
                x = float(row['x'])*METERS_in_NM
                y = float(row['y'])*METERS_in_NM
                if row['pass_no']:
                    pass_no = int(row['pass_no'])
                else:
                    pass_no=0

                metadata = {
                    'cal': float(row['cal']),
                    'sensor': 'RDU103', 
                    'pass_no': pass_no
                    }


                yield timestamp, {Clutter(
                    [x, y], timestamp=timestamp,
                    metadata=metadata, 
                    measurement_model=self.model)}

class CSVReaderPolar(DetectionReader):
    rdp_file: str = Property(doc="File with the radar data.")
    ndim_state: int = Property(default=4)
    pos_mapping: Tuple[int, int] = Property(default=(0, 2))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Kaggle Approach
        measurement_model = CartesianToBearingRange(ndim_state=self.ndim_state, 
                                                 mapping=self.pos_mapping, 
                                                 noise_covar=np.diag([np.radians(0.2), 1]))
        
        self.model=measurement_model


    @BufferedGenerator.generator_method
    def detections_gen(self):
        with open(self.rdp_file, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                if not row['timestamp']:
                    continue

                timestamp = dateutil.parser.parse(row['timestamp'], ignoretz=True)
                rho = float(row['rho'])*METERS_in_NM
                phi = 2*pi - float(row['theta_rad']) + pi/2

                if row['pass_no']:
                    pass_no = int(row['pass_no'])
                else:
                    pass_no=0


                metadata = {
                    'cal': float(row['cal']),
                    'sensor': 'RDU103', 
                    'pass_no': pass_no
                    }

                yield timestamp, {Detection(
                    [Bearing(phi), rho], timestamp=timestamp,
                    # [Bearing(phi), rho, dx, dy], timestamp=timestamp,
                    metadata=metadata, 
                    measurement_model=self.model)}


class ADSBTruthReader(GroundTruthReader):
    adsb_file: str = Property(doc="File with the adsb data.")

    @staticmethod
    def single_ground_truth_reader(adsb_file, isset=True):
        truth = GroundTruthPath()
        with open(adsb_file, newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                alt = float(row['flight_level'])*100
                time = dateutil.parser.parse(row['timestamp'], ignoretz=True)
                if row['target_address'] != "":
                    planename = row['target_address']
                x, y, z = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
                truth.append(GroundTruthState(
                    [x, 0, y, 0, z, 0],
                    timestamp=time,
                    metadata={"id": planename}))
            if isset:
                truth = {truth}
        return truth

    @classmethod
    def multiple_ground_truth_reader(cls, filenames):
        truths = set()
        for filename in filenames:
            truths.add(cls.single_ground_truth_reader(filename, isset=False))
        return truths

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        truths = self.multiple_ground_truth_reader(self.adsb_file)
        yield None, truths


def plot_all(start_time, end_time, all_measurements=None, adsb=None, q_const=None, tracks=None, plot_type='static'):

    if q_const is not None:
        plot_title = f'Kalman Filter, Vconst={q_const}'
    else:
        plot_title = 'Test Data'

    # rdp = RDPReader(rdp_file,
    timestamps = generate_timestamps(start_time, end_time)

    if plot_type == 'animated':
        plotter = AnimatedPlotterly(timestamps, tail_length=0.3, sim_duration=1)

        if all_measurements is not None:
            plotter.plot_measurements(all_measurements,
                                mapping=[0, 2])

        if adsb is not None:
            plotter.plot_ground_truths(adsb, 
                                mapping=[0, 2], 
                                mode='markers', 
                                marker=dict(color='rgba(0, 0, 255, 0.2)',
                                            size=5, 
                                            symbol="square-open")
                                    )
        
        if tracks is not None:
            plotter.plot_tracks(tracks,
                                mapping=[0, 2], 
                                uncertainty=False,
                                marker=dict(color='rgba(57, 255, 20, 1)' ))

        plotter.fig.update_layout(title={'text': plot_title, 
                                    'x': 0.5, 
                                    'xanchor': 'center', 
                                    'yanchor':  'top'})
        # plt.grid()
        plotter.fig.show()

    elif plot_type=='static':
        plotter = Plotter()

        if all_measurements is not None:
            plotter.plot_measurements(all_measurements,
                                    mapping=[0, 2])
        
        if adsb is not None:
            plotter.plot_ground_truths(adsb,
                                    mapping=[0, 2], 
                                    color='purple',
                                    markersize=5, 
                                    marker="s", 
                                    markerfacecolor='none', 
                                    alpha=0.3)
        
        if tracks is not None:
            plotter.plot_tracks(tracks,
                                mapping=[0, 2], 
                                uncertainty=False,
                                color='lime')

        plt.grid()
        plt.title(plot_title)
        plt.show()
    
    return

def group_plots(all_measurements):

    # Group objects by the second of their timestamp
    grouped_objects_sec = defaultdict(set) 
    grouped_objects_pass = defaultdict(set)

    for meas in all_measurements:
    # for meas in dets:
        # Get the second part of the timestamp
        timestamp_s = meas.timestamp.replace(microsecond=0)  # Truncate to second precision
        pass_no = meas.metadata['pass_no']

        grouped_objects_sec[timestamp_s].add(meas)
        grouped_objects_pass[pass_no].add(meas)


    # Convert grouped objects to a set of sets
    grouped_sec = [group for group in grouped_objects_sec.values()]
    grouped_pass = [group for group in grouped_objects_pass.values()]

    return grouped_sec, grouped_pass
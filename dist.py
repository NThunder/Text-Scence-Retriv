#!/usr/bin/env python3
"""
Compute route length statistics from nuScenes tracks.
Add this to your collect_nuscenes_stats.py or run separately.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def parse_args():
    parser = argparse.ArgumentParser(description="Compute route lengths from nuScenes")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max_scenes", type=int, default=None)
    return parser.parse_args()


def get_scene_list(nusc, split):
    if split == "train":
        scene_names = splits.train
    elif split == "val":
        scene_names = splits.val
    else:
        raise ValueError(f"Unknown split {split}")
    return [s for s in nusc.scene if s['name'] in scene_names]


def haversine_distance(pos1, pos2):
    """Calculate distance between two points in meters (assuming UTM or similar)."""
    # For nuScenes, coordinates are in meters, so simple Euclidean is fine
    return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))


def compute_route_statistics(nusc, scenes, min_frames=3):
    """
    Compute route statistics for all objects.
    
    Returns:
        routes: list of dicts with route information
    """
    routes = []
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        sample_token = scene['first_sample_token']
        
        # Dictionary to store tracks: instance_token -> list of positions
        tracks = {}
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            timestamp = sample['timestamp']
            
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                inst_token = ann['instance_token']
                
                # Filter low-quality objects
                if ann['num_lidar_pts'] + ann['num_radar_pts'] < 10:
                    continue
                
                # Get position and velocity
                box = nusc.get_box(ann_token)
                position = box.center.tolist()
                velocity = nusc.box_velocity(box.token),
                
                if inst_token not in tracks:
                    tracks[inst_token] = {
                        'category': box.name,
                        'positions': [],
                        'timestamps': [],
                        'velocities': [],
                        'start_sample': sample_token,
                        'start_time': timestamp,
                    }
                
                tracks[inst_token]['positions'].append(position)
                tracks[inst_token]['timestamps'].append(timestamp)
                tracks[inst_token]['velocities'].append(velocity)
                tracks[inst_token]['end_sample'] = sample_token
                tracks[inst_token]['end_time'] = timestamp
            
            sample_token = sample['next']
        
        # Process all tracks in this scene
        for inst_token, track in tracks.items():
            if len(track['positions']) < min_frames:
                continue
            
            positions = np.array(track['positions'])
            
            # 1. Euclidean distance (straight line)
            start_pos = positions[0][:2]
            end_pos = positions[-1][:2]
            euclidean_distance = np.linalg.norm(end_pos - start_pos)
            
            # 2. Actual traveled distance (sum of segments)
            traveled_distance = 0.0
            segment_distances = []
            for i in range(1, len(positions)):
                seg_dist = np.linalg.norm(positions[i][:2] - positions[i-1][:2])
                traveled_distance += seg_dist
                segment_distances.append(seg_dist)
            
            # 3. Duration in seconds
            duration = (track['end_time'] - track['start_time']) / 1e6
            
            # 4. Average speed
            avg_speed = traveled_distance / duration if duration > 0 else 0.0
            
            # 5. Curvature / straightness ratio (1 = straight line, >1 = curved)
            straightness_ratio = euclidean_distance / traveled_distance if traveled_distance > 0 else 1.0
            
            # 6. Number of direction changes (rough estimate)
            directions = []
            for i in range(1, len(positions)):
                vec = positions[i][:2] - positions[i-1][:2]
                if np.linalg.norm(vec) > 0.1:
                    angle = np.arctan2(vec[1], vec[0])
                    directions.append(angle)
            
            direction_changes = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                if angle_diff > np.pi/6:  # >30 degrees change
                    direction_changes += 1
            
            # 7. Speed variance
            speeds = [np.linalg.norm(v[:2]) for v in track['velocities']]
            speed_variance = np.var(speeds) if len(speeds) > 1 else 0.0
            
            # 8. Determine route type based on shape
            if traveled_distance < 50:
                route_type = "short"
            elif traveled_distance < 200:
                route_type = "medium"
            else:
                route_type = "long"
            
            if straightness_ratio > 0.9:
                route_shape = "straight"
            elif straightness_ratio > 0.7:
                route_shape = "slightly_curved"
            else:
                route_shape = "curvy"
            
            routes.append({
                'instance_token': inst_token,
                'category': track['category'],
                'num_frames': len(track['positions']),
                'duration_seconds': duration,
                'euclidean_distance_m': euclidean_distance,
                'traveled_distance_m': traveled_distance,
                'avg_speed_mps': avg_speed,
                'avg_speed_kmh': avg_speed * 3.6,
                'max_segment_m': max(segment_distances) if segment_distances else 0,
                'straightness_ratio': straightness_ratio,
                'direction_changes': direction_changes,
                'speed_variance': speed_variance,
                'route_type': route_type,
                'route_shape': route_shape,
                'start_pos_x': start_pos[0],
                'start_pos_y': start_pos[1],
                'end_pos_x': end_pos[0],
                'end_pos_y': end_pos[1],
            })
    
    return routes


def compute_route_statistics_from_tracks(tracks_file, output_dir):
    """
    Alternative: compute route statistics from already collected tracks CSV.
    This is faster if you already ran the main collection script.
    """
    df = pd.read_csv(tracks_file)
    
    routes = []
    for inst_token, group in df.groupby('instance_token'):
        if len(group) < 3:
            continue
        
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Positions (assuming you have x,y from translation)
        # Note: This requires that you saved positions in the CSV
        # You may need to adjust based on your actual CSV columns
        
        if 'translation_x' not in group.columns:
            print("Warning: translation columns not found in CSV. Run the main collection script first.")
            return []
        
        positions = np.array([[x, y] for x, y in zip(group['translation_x'], group['translation_y'])])
        
        # Calculate distances
        euclidean_distance = np.linalg.norm(positions[-1] - positions[0])
        
        traveled_distance = 0.0
        for i in range(1, len(positions)):
            traveled_distance += np.linalg.norm(positions[i] - positions[i-1])
        
        duration = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]) / 1e6
        avg_speed = traveled_distance / duration if duration > 0 else 0
        
        routes.append({
            'instance_token': inst_token,
            'category': group['category'].iloc[0],
            'num_frames': len(group),
            'duration_seconds': duration,
            'euclidean_distance_m': euclidean_distance,
            'traveled_distance_m': traveled_distance,
            'avg_speed_mps': avg_speed,
            'straightness_ratio': euclidean_distance / traveled_distance if traveled_distance > 0 else 1.0,
        })
    
    return routes


def generate_route_statistics_summary(routes, output_dir):
    """Generate summary statistics and visualizations."""
    
    # Convert to DataFrame
    df_routes = pd.DataFrame(routes)
    
    # Filter only vehicles (not pedestrians, cones, etc.)
    vehicle_routes = df_routes[df_routes['category'].str.startswith('vehicle.')]
    
    # Statistics by route type
    route_stats = {
        'all_routes': {
            'count': len(df_routes),
            'mean_distance_m': df_routes['traveled_distance_m'].mean(),
            'std_distance_m': df_routes['traveled_distance_m'].std(),
            'median_distance_m': df_routes['traveled_distance_m'].median(),
            'percentile_25_m': df_routes['traveled_distance_m'].quantile(0.25),
            'percentile_75_m': df_routes['traveled_distance_m'].quantile(0.75),
            'percentile_90_m': df_routes['traveled_distance_m'].quantile(0.90),
            'percentile_95_m': df_routes['traveled_distance_m'].quantile(0.95),
            'mean_duration_s': df_routes['duration_seconds'].mean(),
            'median_duration_s': df_routes['duration_seconds'].median(),
            'mean_speed_mps': df_routes['avg_speed_mps'].mean(),
            'mean_straightness': df_routes['straightness_ratio'].mean(),
        },
        'vehicles_only': {
            'count': len(vehicle_routes),
            'mean_distance_m': vehicle_routes['traveled_distance_m'].mean(),
            'std_distance_m': vehicle_routes['traveled_distance_m'].std(),
            'median_distance_m': vehicle_routes['traveled_distance_m'].median(),
            'percentile_25_m': vehicle_routes['traveled_distance_m'].quantile(0.25),
            'percentile_75_m': vehicle_routes['traveled_distance_m'].quantile(0.75),
            'percentile_90_m': vehicle_routes['traveled_distance_m'].quantile(0.90),
            'percentile_95_m': vehicle_routes['traveled_distance_m'].quantile(0.95),
            'mean_duration_s': vehicle_routes['duration_seconds'].mean(),
            'median_duration_s': vehicle_routes['duration_seconds'].median(),
            'mean_speed_mps': vehicle_routes['avg_speed_mps'].mean(),
        },
        'by_route_type': {
            route_type: {
                'count': len(group),
                'mean_distance_m': group['traveled_distance_m'].mean(),
                'median_distance_m': group['traveled_distance_m'].median(),
            }
            for route_type, group in df_routes.groupby('route_type')
        },
        'by_route_shape': {
            shape: {
                'count': len(group),
                'mean_distance_m': group['traveled_distance_m'].mean(),
                'straightness_ratio': group['straightness_ratio'].mean(),
            }
            for shape, group in df_routes.groupby('route_shape')
        },
        'by_category': df_routes['category'].value_counts().to_dict(),
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'route_statistics.json'), 'w') as f:
        json.dump(route_stats, f, indent=2)
    
    # Save full routes data
    df_routes.to_csv(os.path.join(output_dir, 'routes_data.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("ROUTE LENGTH STATISTICS")
    print("="*60)
    print(f"Total routes analyzed: {len(df_routes)}")
    print(f"  - Vehicles only: {len(vehicle_routes)}")
    print(f"\nTraveled Distance (all objects):")
    print(f"  Mean: {route_stats['all_routes']['mean_distance_m']:.1f} m")
    print(f"  Median: {route_stats['all_routes']['median_distance_m']:.1f} m")
    print(f"  Std: {route_stats['all_routes']['std_distance_m']:.1f} m")
    print(f"  90th percentile: {route_stats['all_routes']['percentile_90_m']:.1f} m")
    print(f"  95th percentile: {route_stats['all_routes']['percentile_95_m']:.1f} m")
    print(f"\nDuration:")
    print(f"  Mean: {route_stats['all_routes']['mean_duration_s']:.1f} s")
    print(f"  Median: {route_stats['all_routes']['median_duration_s']:.1f} s")
    print(f"\nRoute Types:")
    for rt, stats in route_stats['by_route_type'].items():
        print(f"  {rt}: {stats['count']} routes, mean {stats['mean_distance_m']:.1f} m")
    
    # Generate recommendations for MetaDrive
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR METADRIVE")
    print("="*60)
    print(f"horizon (seconds): {int(route_stats['vehicles_only']['median_duration_s'])}")
    print(f"  (range: {int(route_stats['vehicles_only']['percentile_25_m']/10)}-{int(route_stats['vehicles_only']['percentile_75_m']/10)} seconds)")
    print(f"destination range: {int(route_stats['vehicles_only']['median_distance_m'])} meters")
    print(f"  (25-75 percentile: {int(route_stats['vehicles_only']['percentile_25_m'])}-{int(route_stats['vehicles_only']['percentile_75_m'])} m)")
    
    return route_stats


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading NuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    scenes = get_scene_list(nusc, args.split)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    
    print(f"Processing {len(scenes)} scenes...")
    routes = compute_route_statistics(nusc, scenes)
    
    print(f"Collected {len(routes)} routes")
    generate_route_statistics_summary(routes, args.output_dir)
    
    # Also save as CSV for further analysis
    df = pd.DataFrame(routes)
    df.to_csv(os.path.join(args.output_dir, f'routes_{args.split}.csv'), index=False)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
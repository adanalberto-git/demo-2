import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

class InfantDataGenerator:
    def __init__(self, output_dir="generated_data"):
        """
        Initialize the data generator
        
        Parameters:
        output_dir: directory where CSV files will be saved
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def generate_infant_data(self, age_months, session_duration_s=300):
        """
        Generate synthetic data for one infant session
        
        Parameters:
        age_months: infant age in months (9-24)
        session_duration_s: session duration in seconds
        
        Returns:
        DataFrame with timestamped gaze and vocalization data
        """
        # Interpolate stats based on age
        age_factor = (age_months - 9) / 15  # 0 to 1 scale from 9 to 24 months
        
        # Vocalization parameters
        voc_rate = 14 + (31.5 - 14) * age_factor  # Vocalizations per session
        voc_duration_mean = 0.653 + (0.896 - 0.653) * age_factor
        voc_duration_std = 0.450 * (1 - age_factor) + 0.263 * age_factor
        
        # Generate timestamps at 30Hz (using integer milliseconds)
        n_samples = int(session_duration_s * 30)
        timestamps = pd.date_range(
            start=datetime(2024,1,1,0,0,0),
            periods=n_samples,
            freq='33ms'  # Using 33ms as approximate 30Hz
        )
        
        # Initialize dataframe
        df = pd.DataFrame(index=timestamps)
        df['age_months'] = age_months
        df['time_s'] = (df.index - df.index[0]).total_seconds()
        
        # Generate gaze data (x,y coordinates normalized -1 to 1)
        df['gaze_x'] = np.random.normal(0, 0.3, len(df))
        df['gaze_y'] = np.random.normal(0, 0.3, len(df))
        
        # Add gaze velocity
        sampling_period = 0.033  # 33ms in seconds
        df['gaze_velocity_x'] = df['gaze_x'].diff() / sampling_period
        df['gaze_velocity_y'] = df['gaze_y'].diff() / sampling_period
        df['gaze_velocity'] = np.sqrt(df['gaze_velocity_x']**2 + df['gaze_velocity_y']**2)
        
        # Generate head orientation (degrees)
        df['head_yaw'] = np.random.normal(0, 15, len(df))
        df['head_pitch'] = np.random.normal(-10, 10, len(df))
        
        # Add head velocity
        df['head_yaw_velocity'] = df['head_yaw'].diff() / sampling_period
        df['head_pitch_velocity'] = df['head_pitch'].diff() / sampling_period
        
        # Generate vocalizations
        n_vocs = int(np.random.normal(voc_rate, voc_rate/4))
        voc_starts = np.sort(np.random.choice(
            df.time_s.values, 
            size=n_vocs, 
            replace=False
        ))
        
        # Add vocalization events and properties
        df['vocalization'] = 0
        df['vocalization_id'] = -1
        df['vocalization_duration'] = 0
        
        for i, start in enumerate(voc_starts):
            duration = np.random.normal(voc_duration_mean, voc_duration_std)
            duration = max(0.1, min(2.0, duration))  # Clip to reasonable range
            
            mask = (df.time_s >= start) & (df.time_s <= start + duration)
            df.loc[mask, 'vocalization'] = 1
            df.loc[mask, 'vocalization_id'] = i
            df.loc[mask, 'vocalization_duration'] = duration
            
        # Add recognizability scores
        recognizability_mean = 0.384 + (1.833 - 0.384) * age_factor
        recognizability_std = 0.431 * (1 - age_factor) + 0.549 * age_factor
        
        df['vocalization_recognizability'] = -1
        for i in range(n_vocs):
            mask = df.vocalization_id == i
            score = np.random.normal(recognizability_mean, recognizability_std)
            score = max(0, min(4, score))  # Clip to 0-4 range
            df.loc[mask, 'vocalization_recognizability'] = score
            
        return df

    def generate_full_dataset(self, n_infants=44, ages=[9,12,15,18,21,24], validate=True):
        """Generate dataset for multiple infants across age groups"""
        dfs = []
        
        # Assign roughly equal numbers of infants to each age group
        infant_ages = []
        for age in ages:
            n_in_group = n_infants // len(ages)
            infant_ages.extend([age] * n_in_group)
        
        # Generate data for each infant
        for i, age in enumerate(infant_ages):
            df = self.generate_infant_data(age)
            df['infant_id'] = i
            dfs.append(df)
            
        full_dataset = pd.concat(dfs, axis=0)
        
        if validate:
            self.validate_dataset(full_dataset)
            
        return full_dataset
    
    def validate_dataset(self, df):
        """Run validation checks on the generated dataset"""
        print("\nValidation Report:")
        print("-----------------")
        print(f"Number of infants: {df.infant_id.nunique()}")
        print(f"Age groups: {sorted(df.age_months.unique())}")
        print(f"Mean vocalizations per session: {df.groupby('infant_id').vocalization_id.max().mean():.1f}")
        
        voc_durations = df[df.vocalization == 1].groupby('vocalization_id').vocalization_duration.first()
        print(f"\nVocalization durations (seconds):")
        print(f"Mean: {voc_durations.mean():.3f}")
        print(f"Std: {voc_durations.std():.3f}")
        
        rec_scores = df[df.vocalization == 1].groupby('vocalization_id').vocalization_recognizability.first()
        print(f"\nRecognizability scores:")
        print(f"Mean: {rec_scores.mean():.3f}")
        print(f"Std: {rec_scores.std():.3f}")
        
        print(f"\nGaze velocity (degrees/s):")
        print(f"Mean: {df.gaze_velocity.mean():.3f}")
        print(f"95th percentile: {df.gaze_velocity.quantile(0.95):.3f}")
    
    def save_dataset(self, df, filename="infant_data.csv"):
        """Save the dataset to a CSV file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath)
        print(f"\nDataset saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        
    def plot_summary(self, df):
        """Generate summary plots of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Vocalization rate by age
        voc_rates = df.groupby(['infant_id', 'age_months']).vocalization_id.max()
        axes[0,0].boxplot([voc_rates[age_months].values 
                          for age_months in sorted(df.age_months.unique())])
        axes[0,0].set_xticklabels(sorted(df.age_months.unique()))
        axes[0,0].set_title('Vocalization Rate by Age')
        axes[0,0].set_xlabel('Age (months)')
        axes[0,0].set_ylabel('Vocalizations per session')
        
        # Plot 2: Recognizability by age
        rec_scores = df[df.vocalization == 1].groupby(['infant_id', 'age_months']).vocalization_recognizability.mean()
        axes[0,1].boxplot([rec_scores[age_months].values 
                          for age_months in sorted(df.age_months.unique())])
        axes[0,1].set_xticklabels(sorted(df.age_months.unique()))
        axes[0,1].set_title('Recognizability by Age')
        axes[0,1].set_xlabel('Age (months)')
        axes[0,1].set_ylabel('Mean recognizability score')
        
        # Plot 3: Gaze position heatmap
        axes[1,0].hist2d(df.gaze_x, df.gaze_y, bins=50)
        axes[1,0].set_title('Gaze Position Heatmap')
        axes[1,0].set_xlabel('Horizontal position')
        axes[1,0].set_ylabel('Vertical position')
        
        # Plot 4: Head orientation heatmap
        axes[1,1].hist2d(df.head_yaw, df.head_pitch, bins=50)
        axes[1,1].set_title('Head Orientation Heatmap')
        axes[1,1].set_xlabel('Yaw (degrees)')
        axes[1,1].set_ylabel('Pitch (degrees)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_plots.png'))
        plt.close()
        print(f"\nSummary plots saved to: {os.path.join(self.output_dir, 'summary_plots.png')}")

# Usage example
if __name__ == "__main__":
    # Initialize generator
    generator = InfantDataGenerator(output_dir="infant_data_output")
    
    # Generate dataset
    dataset = generator.generate_full_dataset(n_infants=44)
    
    # Save to CSV
    generator.save_dataset(dataset)
    
    # Generate summary plots
    generator.plot_summary(dataset)
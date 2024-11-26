# Visual-Vocal Development Analysis

This Python code performs an integrated analysis of visual attention and vocalization 
patterns in infant development across the 9-24 month age range. The analysis includes 
examination of gaze stability, head-gaze coordination, vocalization rates, and the 
relationships between visual and vocal development.

## Features

- Visual attention pattern analysis including gaze stability and head-gaze coordination
- Vocalization capability analysis including rate and recognizability 
- Analysis of relationships between visual attention and vocalizations
- Advanced statistical analyses including mixed effects modeling
- Automated visualization generation
- PDF report generation with detailed findings

## Requirements

The following Python libraries are required:

- numpy
- pandas
- seaborn 
- sklearn
- scipy
- matplotlib
- statsmodels
- reportlab

You can install them using pip:

```bash
pip install numpy pandas seaborn scikit-learn scipy matplotlib statsmodels reportlab
```

## Usage

1. Ensure you have a CSV file named 'infant_data.csv' containing the following columns:
- infant_id
- age_months 
- gaze_velocity
- gaze_velocity_x
- gaze_velocity_y
- gaze_x
- gaze_y
- head_yaw_velocity
- head_pitch_velocity
- vocalization
- vocalization_duration
- vocalization_recognizability

2. Create an output directory named 'infant_data_output' in the same directory as the script:

```bash
mkdir infant_data_output
```

3. Run the script:

```bash
python finalanalysis.py
```

The script will:
- Load and process the data
- Generate various analyses and visualizations
- Create plots saved as PNG files in the infant_data_output directory
- Generate a comprehensive PDF report named 'visual_vocal_analysis_report.pdf'

## Output

The script generates:

1. PNG files for different analyses:
- visual_development_analysis.png
- vocal_development_analysis.png  
- relationships_analysis.png
- advanced_analysis.png

2. A comprehensive PDF report containing:
- Research questions
- Visual attention analysis
- Vocalization analysis  
- Visual-vocal relationships analysis
- Advanced statistical analyses

All output files are saved in the 'infant_data_output' directory.

## Error Handling

The code includes robust error handling and will:
- Print warnings for any analysis steps that fail
- Continue execution when possible
- Provide informative error messages
- Clean up temporary files even if errors occur

## Acknowledgments

This code was developed to analyze infant development data and builds on research in developmental psychology and cognitive science.

## License


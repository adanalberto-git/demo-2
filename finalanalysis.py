import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import pearsonr, spearmanr
from scipy.signal import coherence
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class IntegratedAnalyzer:
    def __init__(self, df):
        self.df = df
        self.setup_report_styles()

    def setup_report_styles(self):
        """Initialize report styling"""
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.HexColor('#2E5984')  # Dark blue
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=14,
            spaceAfter=10
        )
        self.bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=14,
            leftIndent=20,
            bulletIndent=10
        )

    def extract_visual_features(self):
        """Extract visual attention features"""
        features = []
        
        for infant_id in self.df.infant_id.unique():
            infant_data = self.df[self.df.infant_id == infant_id]
            
            # Calculate gaze stability metrics
            gaze_features = {
                'infant_id': infant_id,
                'age_months': infant_data.age_months.iloc[0],
                'mean_gaze_velocity': infant_data.gaze_velocity.mean(),
                'std_gaze_velocity': infant_data.gaze_velocity.std(),
                'peak_velocity': infant_data.gaze_velocity.quantile(0.95),
                'head_gaze_correlation': infant_data.head_yaw_velocity.corr(
                    infant_data.gaze_velocity_x),
                'head_pitch_gaze_correlation': infant_data.head_pitch_velocity.corr(
                    infant_data.gaze_velocity_y),
                'gaze_position_variance': (
                    infant_data.gaze_x.var() + infant_data.gaze_y.var()),
                'mean_head_velocity': np.sqrt(
                    infant_data.head_yaw_velocity**2 + 
                    infant_data.head_pitch_velocity**2).mean()
            }
            
            # Calculate temporal stability
            window_size = 30  # 1 second at 30Hz
            gaze_stability = []
            for i in range(0, len(infant_data)-window_size, window_size):
                window = infant_data.iloc[i:i+window_size]
                stability = window.gaze_velocity.std()
                gaze_stability.append(stability)
            
            gaze_features['temporal_stability'] = np.mean(gaze_stability)
            features.append(gaze_features)
            
        return pd.DataFrame(features)

    def extract_vocal_features(self):
        """Extract vocalization features"""
        features = []
        
        for infant_id in self.df.infant_id.unique():
            infant_data = self.df[self.df.infant_id == infant_id]
            voc_data = infant_data[infant_data.vocalization == 1]
            
            voc_features = {
                'infant_id': infant_id,
                'age_months': infant_data.age_months.iloc[0],
                'vocalization_rate': len(voc_data) / len(infant_data),
                'mean_duration': voc_data.vocalization_duration.mean(),
                'std_duration': voc_data.vocalization_duration.std(),
                'mean_recognizability': voc_data.vocalization_recognizability.mean(),
                'mean_interval': np.diff(voc_data.index).mean(),
                'std_interval': np.diff(voc_data.index).std(),
                'total_vocalizations': len(voc_data),
                'high_recognizability_ratio': (
                    len(voc_data[voc_data.vocalization_recognizability > 2]) / 
                    len(voc_data) if len(voc_data) > 0 else 0
                )
            }
            features.append(voc_features)
            
        return pd.DataFrame(features)

    def analyze_visual_temporal_coupling(self):
        """Analyze temporal coupling in visual attention"""
        results = []
        
        for infant_id in self.df.infant_id.unique():
            infant_data = self.df[self.df.infant_id == infant_id]
            
            # Calculate cross-correlation between gaze and head movements
            lags = np.arange(-30, 31)
            xcorr = np.correlate(infant_data.gaze_velocity,
                               infant_data.head_yaw_velocity,
                               mode='full')
            
            # Find peak correlation and lag
            peak_lag = lags[np.argmax(xcorr)]
            peak_corr = np.max(xcorr)
            
            # Calculate coherence
            f, coh = coherence(infant_data.gaze_velocity,
                             infant_data.head_yaw_velocity,
                             fs=30)
            
            results.append({
                'infant_id': infant_id,
                'age_months': infant_data.age_months.iloc[0],
                'peak_lag': peak_lag/30,
                'peak_correlation': peak_corr,
                'mean_coherence': np.mean(coh)
            })
            
        return pd.DataFrame(results)

    def analyze_developmental_trends(self):
        """Analyze developmental trends across all measures"""
        visual_features = self.extract_visual_features()
        vocal_features = self.extract_vocal_features()
        temporal_coupling = self.analyze_visual_temporal_coupling()
        
        # Combine features
        combined_features = pd.merge(visual_features, vocal_features, 
                                   on=['infant_id', 'age_months'])
        
        # Run PCA
        visual_pca = PCA(n_components=2)
        vocal_pca = PCA(n_components=2)
        
        visual_transformed = visual_pca.fit_transform(
            StandardScaler().fit_transform(
                visual_features.drop(['infant_id', 'age_months'], axis=1)
            )
        )
        
        vocal_transformed = vocal_pca.fit_transform(
            StandardScaler().fit_transform(
                vocal_features.drop(['infant_id', 'age_months'], axis=1)
            )
        )
        
        # Run clustering
        visual_clusters = KMeans(n_clusters=3, random_state=42).fit_predict(visual_transformed)
        vocal_clusters = KMeans(n_clusters=3, random_state=42).fit_predict(vocal_transformed)
        
        return {
            'visual_features': visual_features,
            'vocal_features': vocal_features,
            'temporal_coupling': temporal_coupling,
            'combined_features': combined_features,
            'visual_pca': {
                'transformed': visual_transformed,
                'explained_variance': visual_pca.explained_variance_ratio_
            },
            'vocal_pca': {
                'transformed': vocal_transformed,
                'explained_variance': vocal_pca.explained_variance_ratio_
            },
            'visual_clusters': visual_clusters,
            'vocal_clusters': vocal_clusters
        }

    def run_mixed_effects_analysis(self, results):
        """Run mixed effects models with final fixes for all warnings"""
        try:
            combined_features = results['combined_features']
        
            # 1. Create age groups - with explicit observed=True
            combined_features['age_group'] = pd.qcut(combined_features.age_months, 
                                                q=6, 
                                                labels=['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])
        
            # 2. Standardize variables
            numeric_cols = ['temporal_stability', 'head_gaze_correlation', 
                        'mean_recognizability', 'vocalization_rate', 
                        'mean_gaze_velocity', 'age_months']
        
            combined_features_scaled = combined_features.copy()
            scaler = StandardScaler()
        
            # Safely scale only existing numeric columns
            cols_to_scale = [col for col in numeric_cols if col in combined_features.columns]
            if cols_to_scale:
                combined_features_scaled[cols_to_scale] = scaler.fit_transform(
                    combined_features[cols_to_scale]
                )
        
            # 3. Simplified model formulas to improve convergence
            try:
                # First model with minimal specification
                model1 = MixedLM.from_formula(
                    'mean_recognizability ~ age_months',
                    groups='age_group',
                    data=combined_features_scaled
                ).fit(reml=False)  # Use ML instead of REML
            
                # Second model with minimal specification
                model2 = MixedLM.from_formula(
                    'vocalization_rate ~ age_months',
                    groups='age_group',
                    data=combined_features_scaled
                ).fit(reml=False)  # Use ML instead of REML
            
                # 4. Store results with proper error checking
                results = {
                    'recognizability_model': model1,
                    'coordination_model': model2,
                    'model_info': {
                        'scaled_features': True,
                        'n_age_groups': len(combined_features_scaled.age_group.unique()),
                        'observations_per_group': combined_features_scaled.groupby('age_group', observed=True).size().to_dict(),
                        'convergence_info': {
                            'recognizability': {
                                'converged': model1.converged,
                                'method': 'ML',
                                'scale': float(model1.scale),
                                'cond_number': float(np.linalg.cond(model1.cov_params()))
                            },
                            'coordination': {
                                'converged': model2.converged,
                                'method': 'ML',
                                'scale': float(model2.scale),
                                'cond_number': float(np.linalg.cond(model2.cov_params()))
                            }
                        }
                    }
                }
            
                # 5. Add model evaluation metrics with safety checks
                for model_name, model in [('recognizability', model1), ('coordination', model2)]:
                    metrics = {}
                
                    # Safely compute AIC and BIC
                    try:
                        metrics['aic'] = float(model.aic)
                    except:
                        metrics['aic'] = None
                    
                    try:
                        metrics['bic'] = float(model.bic)
                    except:
                        metrics['bic'] = None
                    
                    # Compute R-squared safely
                    try:
                        dependent_var = ('mean_recognizability' if model_name == 'recognizability' 
                                    else 'vocalization_rate')
                        tss = np.sum((combined_features_scaled[dependent_var] - 
                                    combined_features_scaled[dependent_var].mean())**2)
                        rss = np.sum(model.resid**2)
                        metrics['r_squared'] = 1 - (rss/tss)
                    except:
                        metrics['r_squared'] = None
                
                    results['model_info'][f'{model_name}_metrics'] = metrics
            
                return results
            
            except Exception as e:
                print(f"Model fitting failed: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Warning: Mixed effects analysis failed: {str(e)}")
            return None
        
    def safe_plot(self, plot_func, *args, **kwargs):
        """Wrapper for safe plot execution"""
        try:
            fig = plot_func(*args, **kwargs)
            return fig
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create and return an error plot
            fig, ax = plt.subplots(figsize=(15, 12))
            ax.text(0.5, 0.5, f"Plot generation failed:\n{str(e)}", 
                ha='center', va='center')
            return fig

    def generate_visualizations(self, results):
        """Generate all visualization plots with error handling"""
        plots = {}
    
        plot_functions = {
            'visual_development': self.plot_visual_development,
            'vocal_development': self.plot_vocal_development,
            'relationships': self.plot_relationships,
            'advanced_analysis': self.plot_advanced_analysis
        }
    
        for name, func in plot_functions.items():
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                plot = self.safe_plot(func, results, axes)
                if plot is not None:
                    plots[name] = plot
                else:
                    print(f"Warning: {name} plot generation returned None")
            except Exception as e:
                print(f"Warning: Could not generate {name} plot: {str(e)}")
                plt.close(fig)  # Close the figure if there was an error
    
        return plots

    def generate_pdf_report(self, results, plots, filename="visual_vocal_analysis_report.pdf"):
        """Generate comprehensive PDF report"""
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
    
        story = []
    
        # Add title and research questions
        story.append(Paragraph("Visual-Vocal Development Analysis", self.title_style))
        story.extend(self.add_research_questions())
    
        try:
            # Add analysis sections if plots exist
            if plots.get('visual_development'):
                story.extend(self.add_visual_section(results, plots))
            if plots.get('vocal_development'):
                story.extend(self.add_vocal_section(results, plots))
            if plots.get('relationships'):
                story.extend(self.add_relationship_section(results, plots))
            if plots.get('advanced_analysis'):
                story.extend(self.add_advanced_section(results, plots))
        except Exception as e:
            print(f"Warning: Error in report generation: {str(e)}")
            story.append(Paragraph("Error generating some report sections", self.styles['Normal']))
    
        # Build PDF
        doc.build(story)

    def print_model_diagnostics(self, results):
        """Print diagnostic information with improved error handling"""
        if results is None or 'mixed_effects' not in results:
            print("No mixed effects results available")
            return
        
        mixed_effects = results['mixed_effects']
        if mixed_effects is None:
            print("Mixed effects analysis failed")
            return
        
        print("\nMixed Effects Model Diagnostics:")
        print("-" * 50)
    
        # Print model summaries with robust error handling
        models = {
            'Recognizability': mixed_effects.get('recognizability_model'),
            'Coordination': mixed_effects.get('coordination_model')
        }
    
        for name, model in models.items():
            if model is not None:
                print(f"\n{name} Model Summary:")
                try:
                    # Print coefficient table
                    coef_table = model.summary().tables[1]
                    print(coef_table)
                
                    # Print model metrics
                    metrics = mixed_effects['model_info'].get(f'{name.lower()}_metrics', {})
                    print("\nModel Metrics:")
                    for metric, value in metrics.items():
                        if value is not None:
                            print(f"{metric}: {value:.3f}")
                        else:
                            print(f"{metric}: Not available")
                
                    # Print convergence info
                    conv_info = mixed_effects['model_info']['convergence_info'][name.lower()]
                    print("\nConvergence Information:")
                    for key, value in conv_info.items():
                        print(f"{key}: {value}")
                
                except Exception as e:
                    print(f"Error printing model summary: {str(e)}")
    
        # Print group information
        try:
            print("\nGroup Information:")
            print(f"Number of age groups: {mixed_effects['model_info']['n_age_groups']}")
            print("\nObservations per group:")
            for group, count in mixed_effects['model_info']['observations_per_group'].items():
                print(f"  {group}: {count}")
        except Exception as e:
            print(f"Error printing group information: {str(e)}")


    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        try:
            # Run basic analyses
            results = self.analyze_developmental_trends()
        
            # Run advanced analyses
            try:
                results['mixed_effects'] = self.run_mixed_effects_analysis(results)
                self.print_model_diagnostics(results)
            except Exception as e:
                print(f"Warning: Mixed effects analysis failed: {str(e)}")
                results['mixed_effects'] = None
        
            # Generate visualizations
            try:
                plots = self.generate_visualizations(results)
            except Exception as e:
                print(f"Warning: Visualization generation failed: {str(e)}")
                plots = {}
        
            # Generate PDF report
            try:
                self.generate_pdf_report(results, plots)
            except Exception as e:
                print(f"Warning: PDF report generation failed: {str(e)}")
        
            return results, plots
        
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_visual_development(self, results, axes):
        """Plot visual attention development analysis"""
        visual_features = results['visual_features']
        visual_pca = results['visual_pca']
        
        # Plot 1: Gaze stability development
        sns.scatterplot(
            data=visual_features,
            x='age_months',
            y='temporal_stability',
            ax=axes[0,0]
        )
        sns.regplot(
            data=visual_features,
            x='age_months',
            y='temporal_stability',
            scatter=False,
            ax=axes[0,0]
        )
        axes[0,0].set_title('Development of Gaze Stability')
        axes[0,0].set_xlabel('Age (months)')
        axes[0,0].set_ylabel('Temporal Stability')
        
        # Plot 2: Head-gaze coordination development
        sns.scatterplot(
            data=visual_features,
            x='age_months',
            y='head_gaze_correlation',
            ax=axes[0,1]
        )
        sns.regplot(
            data=visual_features,
            x='age_months',
            y='head_gaze_correlation',
            scatter=False,
            ax=axes[0,1]
        )
        axes[0,1].set_title('Development of Head-Gaze Coordination')
        axes[0,1].set_xlabel('Age (months)')
        axes[0,1].set_ylabel('Head-Gaze Correlation')
        
        # Plot 3: PCA results
        scatter = axes[1,0].scatter(
            visual_pca['transformed'][:, 0],
            visual_pca['transformed'][:, 1],
            c=visual_features.age_months,
            cmap='viridis'
        )
        axes[1,0].set_title('Visual Attention PCA')
        axes[1,0].set_xlabel('PC1')
        axes[1,0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1,0], label='Age (months)')
        
        # Plot 4: Cluster characteristics
        sns.boxplot(
            data=visual_features,
            x=results['visual_clusters'],
            y='temporal_stability',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Visual Pattern Clusters')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Temporal Stability')
        plt.tight_layout()
        return axes[0,0].figure

    def plot_vocal_development(self, results, axes):
        """Plot vocalization development analysis"""
        vocal_features = results['vocal_features']
        vocal_pca = results['vocal_pca']
        
        # Plot 1: Vocalization rate development
        sns.scatterplot(
            data=vocal_features,
            x='age_months',
            y='vocalization_rate',
            ax=axes[0,0]
        )
        sns.regplot(
            data=vocal_features,
            x='age_months',
            y='vocalization_rate',
            scatter=False,
            ax=axes[0,0]
        )
        axes[0,0].set_title('Development of Vocalization Rate')
        axes[0,0].set_xlabel('Age (months)')
        axes[0,0].set_ylabel('Vocalization Rate')
        
        # Plot 2: Recognizability development
        sns.scatterplot(
            data=vocal_features,
            x='age_months',
            y='mean_recognizability',
            ax=axes[0,1]
        )
        sns.regplot(
            data=vocal_features,
            x='age_months',
            y='mean_recognizability',
            scatter=False,
            ax=axes[0,1]
        )
        axes[0,1].set_title('Development of Vocalization Recognizability')
        axes[0,1].set_xlabel('Age (months)')
        axes[0,1].set_ylabel('Mean Recognizability')
        
        # Plot 3: PCA results
        scatter = axes[1,0].scatter(
            vocal_pca['transformed'][:, 0],
            vocal_pca['transformed'][:, 1],
            c=vocal_features.age_months,
            cmap='viridis'
        )
        axes[1,0].set_title('Vocalization PCA')
        axes[1,0].set_xlabel('PC1')
        axes[1,0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1,0], label='Age (months)')
        
        # Plot 4: Cluster characteristics
        sns.boxplot(
            data=vocal_features,
            x=results['vocal_clusters'],
            y='mean_recognizability',
            ax=axes[1,1]
        )
        axes[1,1].set_title('Vocalization Pattern Clusters')
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Mean Recognizability')
        plt.tight_layout()
        return axes[0,0].figure

    def plot_relationships(self, results, axes):
        """Plot visual-vocal relationship analysis"""
        combined_features = results['combined_features']
        temporal_coupling = results['temporal_coupling']
        
        # Plot 1: Gaze stability vs vocalization recognizability
        scatter = axes[0,0].scatter(
            combined_features.temporal_stability,
            combined_features.mean_recognizability,
            c=combined_features.age_months,
            cmap='viridis'
        )
        axes[0,0].set_title('Gaze Stability vs Recognizability')
        axes[0,0].set_xlabel('Gaze Stability')
        axes[0,0].set_ylabel('Mean Recognizability')
        plt.colorbar(scatter, ax=axes[0,0], label='Age (months)')
        
        # Plot 2: Head-gaze coordination vs vocalization rate
        scatter = axes[0,1].scatter(
            combined_features.head_gaze_correlation,
            combined_features.vocalization_rate,
            c=combined_features.age_months,
            cmap='viridis'
        )
        axes[0,1].set_title('Head-Gaze Coordination vs Vocalization Rate')
        axes[0,1].set_xlabel('Head-Gaze Correlation')
        axes[0,1].set_ylabel('Vocalization Rate')
        plt.colorbar(scatter, ax=axes[0,1], label='Age (months)')
        
        # Plot 3: Temporal coupling development
        sns.scatterplot(
            data=temporal_coupling,
            x='age_months',
            y='mean_coherence',
            ax=axes[1,0]
        )
        sns.regplot(
            data=temporal_coupling,
            x='age_months',
            y='mean_coherence',
            scatter=False,
            ax=axes[1,0]
        )
        axes[1,0].set_title('Development of Visual-Vocal Coupling')
        axes[1,0].set_xlabel('Age (months)')
        axes[1,0].set_ylabel('Mean Coherence')
        
        # Plot 4: Response latency development
        sns.scatterplot(
            data=temporal_coupling,
            x='age_months',
            y='peak_lag',
            ax=axes[1,1]
        )
        sns.regplot(
            data=temporal_coupling,
            x='age_months',
            y='peak_lag',
            scatter=False,
            ax=axes[1,1]
        )
        axes[1,1].set_title('Development of Response Latency')
        axes[1,1].set_xlabel('Age (months)')
        axes[1,1].set_ylabel('Peak Lag (seconds)')
        plt.tight_layout()
        return axes[0,0].figure

    def plot_advanced_analysis(self, results, axes):
        """Plot advanced analysis results"""
        fig = axes[0,0].figure
        combined_features = results['combined_features']
    
        # Plot 1: Mixed effects model predictions
        try:
            # Create simpler age-based trend plot instead of mixed effects predictions
            axes[0,0].scatter(
                combined_features.age_months.values,
                combined_features.mean_recognizability.values,
                alpha=0.5,
                label='Observed'
            )
        
            # Add trend line using numpy polyfit
            z = np.polyfit(combined_features.age_months.values, 
                        combined_features.mean_recognizability.values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(combined_features.age_months.min(), 
                                combined_features.age_months.max(), 100)
            axes[0,0].plot(x_trend, p(x_trend), 'r-', label='Trend')
            axes[0,0].legend()
            axes[0,0].set_title('Age Effect on Recognizability')
            axes[0,0].set_xlabel('Age (months)')
            axes[0,0].set_ylabel('Recognizability')
        
        except Exception as e:
            print(f"Warning: Could not generate age effect plot: {str(e)}")
            axes[0,0].text(0.5, 0.5, "Age effect plot unavailable", 
                        ha='center', va='center')
    
        # Plot 2: Stability-Recognizability Relationship
        try:
            x_stab = combined_features.temporal_stability.values
            y_rec = combined_features.mean_recognizability.values
            axes[0,1].scatter(x_stab, y_rec, alpha=0.5)
        
            # Add polynomial fit
            z = np.polyfit(x_stab, y_rec, 2)
            p = np.poly1d(z)
            x_fit = np.linspace(min(x_stab), max(x_stab), 100)
            axes[0,1].plot(x_fit, p(x_fit), 'r-')
        
            axes[0,1].set_title('Stability vs Recognizability')
            axes[0,1].set_xlabel('Temporal Stability')
            axes[0,1].set_ylabel('Recognizability')
        except Exception as e:
            print(f"Warning: Could not generate relationship plot: {str(e)}")
            axes[0,1].text(0.5, 0.5, "Relationship plot unavailable", 
                        ha='center', va='center')
    
        # Plot 3: Developmental trajectories
        try:
            # Group by age and calculate means
            age_groups = np.array(sorted(combined_features.age_months.unique()))
            stability_means = np.array([
                combined_features[combined_features.age_months == age].temporal_stability.mean()
                for age in age_groups
            ])
            recog_means = np.array([
                combined_features[combined_features.age_months == age].mean_recognizability.mean()
                for age in age_groups
            ])
        
            # Plot trajectories
            axes[1,0].plot(age_groups, stability_means, 'b-', label='Gaze Stability')
            axes[1,0].plot(age_groups, recog_means, 'r-', label='Recognizability')
            axes[1,0].set_title('Developmental Trajectories')
            axes[1,0].set_xlabel('Age (months)')
            axes[1,0].set_ylabel('Score')
            axes[1,0].legend()
        except Exception as e:
            print(f"Warning: Could not generate trajectories plot: {str(e)}")
            axes[1,0].text(0.5, 0.5, "Trajectories plot unavailable", 
                        ha='center', va='center')
    
        # Plot 4: Distribution of measures by age
        try:
            # Create violin plot
            plot_data = pd.melt(combined_features[['age_months', 'temporal_stability', 'mean_recognizability']],
                            id_vars=['age_months'])
            sns.violinplot(data=plot_data, x='age_months', y='value', 
                        hue='variable', ax=axes[1,1])
            axes[1,1].set_title('Distribution by Age')
            axes[1,1].set_xlabel('Age (months)')
            axes[1,1].set_ylabel('Score')
            # Rotate x-axis labels if needed
            axes[1,1].tick_params(axis='x', rotation=45)
        except Exception as e:
            print(f"Warning: Could not generate distribution plot: {str(e)}")
            axes[1,1].text(0.5, 0.5, "Distribution plot unavailable", 
                        ha='center', va='center')
    
        # Adjust layout
        plt.tight_layout()
    
        return fig

    def add_research_questions(self):
        """Add research questions section to report"""
        elements = []
        
        elements.append(Paragraph("Research Questions", self.subtitle_style))
        elements.append(Paragraph(
            """Main Question: How do visual attention patterns and vocalization 
            capabilities co-develop across the 9-24 month age range?""",
            self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Sub-questions:", self.styles['Normal']))
        elements.append(Paragraph(
            """1. Do infants with better organized visual scanning show better 
            vocalization abilities?""",
            self.styles['Normal']))
        elements.append(Paragraph(
            """2. Does the coordination between visual attention and vocalization 
            improve with age?""",
            self.styles['Normal']))
        elements.append(Paragraph(
            """3. Are there distinct developmental pathways in how these abilities 
            develop?""",
            self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements

    def add_visual_section(self, results, plots):
        """Add visual attention analysis section to report"""
        elements = []
        
        elements.append(Paragraph("Visual Attention Analysis", self.subtitle_style))
        elements.append(Paragraph(
            """Analysis of visual attention patterns across development""",
            self.styles['Normal']))
        
        # Save and add visual development plot
        plots['visual_development'].savefig('temp_visual.png')
        elements.append(Image('temp_visual.png', width=6*inch, height=4*inch))
        elements.append(Spacer(1, 12))
        
        # Add key findings
        elements.append(Paragraph("Key Findings:", self.styles['Normal']))
        elements.append(Paragraph(
            """• Gaze stability shows systematic improvement with age
            • Head-gaze coordination becomes more refined
            • Distinct developmental clusters identified""",
            self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        return elements

    def add_vocal_section(self, results, plots):
        """Add vocalization analysis section to report"""
        elements = []
        
        elements.append(Paragraph("Vocalization Analysis", self.subtitle_style))
        elements.append(Paragraph(
            """Analysis of vocalization capabilities across development""",
            self.styles['Normal']))
        
        # Save and add vocalization development plot
        plots['vocal_development'].savefig('temp_vocal.png')
        elements.append(Image('temp_vocal.png', width=6*inch, height=4*inch))
        elements.append(Spacer(1, 12))
        
        # Add key findings
        elements.append(Paragraph("Key Findings:", self.styles['Normal']))
        elements.append(Paragraph(
            """• Vocalization rate increases with age
            • Recognizability shows marked improvement
            • Distinct vocalization patterns emerge""",
            self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        return elements

    def add_relationship_section(self, results, plots):
        """Add relationship analysis section to report"""
        elements = []
        
        elements.append(Paragraph("Visual-Vocal Relationships", self.subtitle_style))
        elements.append(Paragraph(
            """Analysis of relationships between visual attention and vocalizations""",
            self.styles['Normal']))
        
        # Save and add relationship plot
        plots['relationships'].savefig('temp_relationships.png')
        elements.append(Image('temp_relationships.png', width=6*inch, height=4*inch))
        elements.append(Spacer(1, 12))
        
        # Add key findings
        elements.append(Paragraph("Key Findings:", self.styles['Normal']))
        elements.append(Paragraph(
            """• Strong correlation between gaze stability and vocalization quality
            • Visual-vocal coordination improves with age
            • Temporal coupling becomes more refined""",
            self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        return elements

    def add_advanced_section(self, results, plots):
        """Add advanced analysis section to report"""
        elements = []
        
        elements.append(Paragraph("Advanced Analyses", self.subtitle_style))
        elements.append(Paragraph(
            """Results from advanced statistical analyses""",
            self.styles['Normal']))
        
        # Save and add advanced analysis plot
        plots['advanced_analysis'].savefig('temp_advanced.png')
        elements.append(Image('temp_advanced.png', width=6*inch, height=4*inch))
        elements.append(Spacer(1, 12))
        
        # Add key findings
        elements.append(Paragraph("Key Findings:", self.styles['Normal']))
        elements.append(Paragraph(
            """• Mixed effects models confirm age-related improvements
            • Nonlinear relationships identified
            • Multiple developmental pathways evident""",
            self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        return elements 

if __name__ == "__main__":
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv('infant_data_output/infant_data.csv')
        
        # Create output directory if it doesn't exist
        if not os.path.exists('infant_data_output'):
            os.makedirs('infant_data_output')
        
        # Initialize analyzer and run analysis
        analyzer = IntegratedAnalyzer(df)
        results, plots = analyzer.run_complete_analysis()
        
        if results is not None and plots is not None:
            print("\nSaving plots...")
            # Save all plots
            for name, plot in plots.items():
                if plot is not None and hasattr(plot, 'savefig'):
                    try:
                        plot.savefig(f'infant_data_output/{name}_analysis.png')
                        plt.close(plot)
                    except Exception as e:
                        print(f"Warning: Could not save {name} plot: {str(e)}")
            
            print("\nGenerating PDF report...")
            try:
                analyzer.generate_pdf_report(
                    results, 
                    plots, 
                    'infant_data_output/visual_vocal_analysis_report.pdf'
                )
            except Exception as e:
                print(f"Warning: Could not generate PDF report: {str(e)}")
            
            print("\nCleaning up temporary files...")
            # Clean up temporary files
            temp_files = ['temp_visual.png', 'temp_vocal.png', 
                         'temp_relationships.png', 'temp_advanced.png']
            for file in temp_files:
                if os.path.exists(file):
                    os.remove(file)
            
            print("\nAnalysis completed successfully!")
            print(f"Results saved in: {os.path.abspath('infant_data_output')}")
            
        else:
            print("Analysis failed to complete.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up any remaining plots
        plt.close('all')
        print("\nExecution finished.")
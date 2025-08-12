#!/usr/bin/env python3
"""
BioPulse Guardian - Flask Web Application

Main Flask application for the BioPulse Guardian health monitoring system.
Provides web interface for biometric data visualization and health analytics.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
import os
import yaml
import logging
from werkzeug.exceptions import NotFound

# Import custom modules
from features import FeatureExtractor
from rules import HealthRulesEngine
from report import ReportGenerator
from ai_analyzer import AIAnalyzer
from etl import ETLPipeline

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        return {
            'database': {'host': 'localhost', 'port': 5432},
            'ai_models': {'threshold': 0.8},
            'health_rules': {'enabled': True},
            'web': {'host': '0.0.0.0', 'port': 5000, 'debug': False}
        }

config = load_config()

# Initialize components
feature_extractor = FeatureExtractor(config)
rules_engine = HealthRulesEngine(config)
report_generator = ReportGenerator(config)
ai_analyzer = AIAnalyzer(config)
etl_pipeline = ETLPipeline(config)

# Routes
@app.route('/')
def dashboard():
    """Main dashboard view"""
    try:
        # Get latest health metrics
        metrics = get_latest_metrics()
        alerts = rules_engine.check_health_rules(metrics)
        
        return render_template('dashboard.html', 
                             metrics=metrics, 
                             alerts=alerts,
                             timestamp=datetime.now())
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('error.html', error="Dashboard unavailable"), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for latest metrics"""
    try:
        metrics = get_latest_metrics()
        return jsonify({
            'success': True,
            'data': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"API metrics error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for AI analysis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Extract features
        features = feature_extractor.extract(data)
        
        # Run AI analysis
        analysis = ai_analyzer.analyze(features)
        
        # Check rules
        alerts = rules_engine.check_health_rules(features)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reports')
def reports():
    """Reports view"""
    try:
        reports = report_generator.get_available_reports()
        return render_template('reports.html', reports=reports)
    except Exception as e:
        logger.error(f"Reports error: {e}")
        return render_template('error.html', error="Reports unavailable"), 500

@app.route('/reports/<report_id>')
def view_report(report_id):
    """View specific report"""
    try:
        report = report_generator.generate_report(report_id)
        return render_template('report_view.html', report=report)
    except Exception as e:
        logger.error(f"Report view error: {e}")
        return render_template('error.html', error="Report not found"), 404

@app.route('/settings')
def settings():
    """Settings view"""
    return render_template('settings.html', config=config)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        new_settings = request.get_json()
        # Validate and update settings
        # Note: In production, implement proper validation
        config.update(new_settings)
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully'
        })
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Helper functions
def get_latest_metrics():
    """Get latest health metrics from data source"""
    # Mock data for demonstration
    # In production, this would fetch from database
    return {
        'heart_rate': 72,
        'blood_pressure': {'systolic': 120, 'diastolic': 80},
        'temperature': 98.6,
        'oxygen_saturation': 98,
        'activity_level': 'moderate',
        'timestamp': datetime.now().isoformat()
    }

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', error="Internal server error"), 500

# Application factory
def create_app(config_override=None):
    """Application factory for testing"""
    if config_override:
        app.config.update(config_override)
    return app

# Main execution
if __name__ == '__main__':
    logger.info("Starting BioPulse Guardian web application")
    
    # Get web configuration
    web_config = config.get('web', {})
    host = web_config.get('host', '0.0.0.0')
    port = web_config.get('port', 5000)
    debug = web_config.get('debug', False)
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )

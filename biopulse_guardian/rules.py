#!/usr/bin/env python3
"""
BioPulse Guardian Health Monitoring Rules Engine
Implements rule-based health monitoring and alert system
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import yaml
import logging

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class HealthRule:
    """Individual health monitoring rule"""
    
    def __init__(self, rule_id: str, name: str, metric: str, 
                 condition: str, threshold: float, alert_level: AlertLevel):
        self.rule_id = rule_id
        self.name = name
        self.metric = metric
        self.condition = condition  # 'gt', 'lt', 'eq', 'ne'
        self.threshold = threshold
        self.alert_level = alert_level
        self.active = True
        self.created_at = datetime.now()
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if rule condition is met"""
        if not self.active:
            return False
            
        if self.condition == 'gt':
            return value > self.threshold
        elif self.condition == 'lt':
            return value < self.threshold
        elif self.condition == 'eq':
            return value == self.threshold
        elif self.condition == 'ne':
            return value != self.threshold
        elif self.condition == 'gte':
            return value >= self.threshold
        elif self.condition == 'lte':
            return value <= self.threshold
        else:
            return False

class RulesEngine:
    """Health monitoring rules engine"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.rules: List[HealthRule] = []
        self.alerts: List[Dict] = []
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._initialize_default_rules()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for rules engine"""
        return {
            'rules': {
                'heart_rate': {
                    'normal_range': {'min': 60, 'max': 100},
                    'critical_range': {'min': 40, 'max': 180}
                },
                'blood_pressure': {
                    'normal_systolic': {'min': 90, 'max': 140},
                    'normal_diastolic': {'min': 60, 'max': 90},
                    'critical_systolic': {'min': 70, 'max': 200},
                    'critical_diastolic': {'min': 40, 'max': 120}
                },
                'temperature': {
                    'normal_range': {'min': 36.0, 'max': 37.5},
                    'fever_threshold': 38.0,
                    'hypothermia_threshold': 35.0
                },
                'oxygen_saturation': {
                    'normal_min': 95,
                    'critical_min': 85
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_default_rules(self):
        """Initialize default health monitoring rules"""
        rules_config = self.config.get('rules', {})
        
        # Heart rate rules
        hr_config = rules_config.get('heart_rate', {})
        self.add_rule(HealthRule(
            'HR001', 'High Heart Rate', 'heart_rate', 'gt', 
            hr_config.get('normal_range', {}).get('max', 100), AlertLevel.WARNING
        ))
        self.add_rule(HealthRule(
            'HR002', 'Low Heart Rate', 'heart_rate', 'lt',
            hr_config.get('normal_range', {}).get('min', 60), AlertLevel.WARNING
        ))
        self.add_rule(HealthRule(
            'HR003', 'Critical High Heart Rate', 'heart_rate', 'gt',
            hr_config.get('critical_range', {}).get('max', 180), AlertLevel.CRITICAL
        ))
        
        # Blood pressure rules
        bp_config = rules_config.get('blood_pressure', {})
        self.add_rule(HealthRule(
            'BP001', 'High Systolic BP', 'blood_pressure_systolic', 'gt',
            bp_config.get('normal_systolic', {}).get('max', 140), AlertLevel.WARNING
        ))
        self.add_rule(HealthRule(
            'BP002', 'High Diastolic BP', 'blood_pressure_diastolic', 'gt',
            bp_config.get('normal_diastolic', {}).get('max', 90), AlertLevel.WARNING
        ))
        
        # Temperature rules
        temp_config = rules_config.get('temperature', {})
        self.add_rule(HealthRule(
            'TEMP001', 'Fever', 'temperature', 'gte',
            temp_config.get('fever_threshold', 38.0), AlertLevel.WARNING
        ))
        self.add_rule(HealthRule(
            'TEMP002', 'Hypothermia', 'temperature', 'lt',
            temp_config.get('hypothermia_threshold', 35.0), AlertLevel.CRITICAL
        ))
        
        # Oxygen saturation rules
        o2_config = rules_config.get('oxygen_saturation', {})
        self.add_rule(HealthRule(
            'O2001', 'Low Oxygen Saturation', 'oxygen_saturation', 'lt',
            o2_config.get('normal_min', 95), AlertLevel.WARNING
        ))
        self.add_rule(HealthRule(
            'O2002', 'Critical Low Oxygen', 'oxygen_saturation', 'lt',
            o2_config.get('critical_min', 85), AlertLevel.CRITICAL
        ))
        
        self.logger.info(f"Initialized {len(self.rules)} default rules")
    
    def add_rule(self, rule: HealthRule):
        """Add a new monitoring rule"""
        self.rules.append(rule)
        self.logger.info(f"Added rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID"""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                removed_rule = self.rules.pop(i)
                self.logger.info(f"Removed rule: {removed_rule.name}")
                return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[HealthRule]:
        """Get a rule by ID"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Dict]:
        """Evaluate all rules against provided metrics"""
        triggered_alerts = []
        
        for rule in self.rules:
            if rule.metric in metrics:
                value = metrics[rule.metric]
                if rule.evaluate(value):
                    alert = self._create_alert(rule, value)
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
                    self.logger.warning(f"Alert triggered: {alert['message']}")
        
        return triggered_alerts
    
    def _create_alert(self, rule: HealthRule, value: float) -> Dict:
        """Create an alert from a triggered rule"""
        return {
            'alert_id': f"ALERT_{rule.rule_id}_{int(datetime.now().timestamp())}",
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'metric': rule.metric,
            'value': value,
            'threshold': rule.threshold,
            'condition': rule.condition,
            'alert_level': rule.alert_level.value,
            'message': f"{rule.name}: {rule.metric} = {value} (threshold: {rule.threshold})",
            'timestamp': datetime.now().isoformat()
        }
    
    def get_active_alerts(self, hours_back: int = 24) -> List[Dict]:
        """Get recent alerts within specified hours"""
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        
        recent_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_rule_summary(self) -> Dict:
        """Get summary of all rules"""
        active_rules = sum(1 for rule in self.rules if rule.active)
        inactive_rules = sum(1 for rule in self.rules if not rule.active)
        
        by_level = {}
        for rule in self.rules:
            level = rule.alert_level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            'total_rules': len(self.rules),
            'active_rules': active_rules,
            'inactive_rules': inactive_rules,
            'rules_by_level': by_level,
            'total_alerts': len(self.alerts)
        }
    
    def toggle_rule(self, rule_id: str) -> bool:
        """Toggle rule active/inactive status"""
        rule = self.get_rule(rule_id)
        if rule:
            rule.active = not rule.active
            status = 'activated' if rule.active else 'deactivated'
            self.logger.info(f"Rule {rule.name} {status}")
            return True
        return False
    
    def clear_old_alerts(self, days_old: int = 7):
        """Clear alerts older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        initial_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']).timestamp() >= cutoff_time
        ]
        
        cleared_count = initial_count - len(self.alerts)
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} old alerts")

def main():
    """Main execution function for testing"""
    rules_engine = RulesEngine()
    
    # Test with sample metrics
    sample_metrics = {
        'heart_rate': 110,  # High
        'blood_pressure_systolic': 150,  # High
        'blood_pressure_diastolic': 85,
        'temperature': 38.5,  # Fever
        'oxygen_saturation': 92  # Low
    }
    
    print("\n=== BioPulse Guardian Rules Engine Test ===")
    print(f"Rules Summary: {rules_engine.get_rule_summary()}")
    
    alerts = rules_engine.evaluate_metrics(sample_metrics)
    print(f"\nTriggered {len(alerts)} alerts:")
    
    for alert in alerts:
        print(f"- {alert['alert_level'].upper()}: {alert['message']}")

if __name__ == "__main__":
    main()

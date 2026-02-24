import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class AlertSystem:
    def __init__(self, config_file="config/alert_config.json"):
        self.config = self.load_config(config_file)
        self.setup_logging()
    
    def load_config(self, config_file):
        """Load alert configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "email_alerts": True,
                "sms_alerts": False,
                "min_confidence": 0.8,
                "email_settings": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "alerts@company.com",
                    "sender_password": "your_password",
                    "recipient_emails": ["security@company.com"]
                }
            }
    
    def setup_logging(self):
        """Setup logging for alert system"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename='logs/security_alerts.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def send_email_alert(self, attack_details):
        """Send email alert for detected attacks"""
        if not self.config["email_alerts"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email_settings"]["sender_email"]
            msg['To'] = ", ".join(self.config["email_settings"]["recipient_emails"])
            msg['Subject'] = f"SECURITY ALERT: {attack_details['attack_type']} Detected"
            
            body = f"""
            Security Alert Notification
            
            Attack Type: {attack_details['attack_type']}
            Source IP: {attack_details.get('source_ip', 'Unknown')}
            Timestamp: {attack_details.get('timestamp', datetime.now())}
            Confidence: {attack_details.get('confidence', 'N/A')}
            
            Recommended Actions:
            - Investigate source IP
            - Check network logs
            - Update firewall rules if necessary
            
            This is an automated alert from the Network Anomaly Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(
                self.config["email_settings"]["smtp_server"],
                self.config["email_settings"]["smtp_port"]
            )
            server.starttls()
            server.login(
                self.config["email_settings"]["sender_email"],
                self.config["email_settings"]["sender_password"]
            )
            text = msg.as_string()
            server.sendmail(
                self.config["email_settings"]["sender_email"],
                self.config["email_settings"]["recipient_emails"],
                text
            )
            server.quit()
            
            logging.info(f"Email alert sent for {attack_details['attack_type']}")
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def log_alert(self, attack_details):
        """Log security alert to file and console"""
        log_message = (
            f"ALERT - Type: {attack_details['attack_type']}, "
            f"Source: {attack_details.get('source_ip', 'Unknown')}, "
            f"Confidence: {attack_details.get('confidence', 'N/A')}"
        )
        
        logging.info(log_message)
        print(f"SECURITY ALERT: {log_message}")
    
    def trigger_alert(self, prediction, confidence, features=None):
        """Main method to trigger alerts based on predictions"""
        if confidence < self.config["min_confidence"]:
            return
        
        if prediction != 'BENIGN':
            attack_details = {
                'attack_type': prediction,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'source_ip': features.get('source_ip', 'Unknown') if features else 'Unknown'
            }
            
            # Log the alert
            self.log_alert(attack_details)
            
            # Send email alert
            if self.config["email_alerts"]:
                self.send_email_alert(attack_details)
            
            return attack_details

# Singleton instance
alert_system = AlertSystem()
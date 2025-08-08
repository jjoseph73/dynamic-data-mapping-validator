# security/setup-security.sh - Security Setup Script

#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Generate secure random passwords
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

# Generate SSL certificates
generate_ssl_certs() {
    log "Generating SSL certificates..."
    
    mkdir -p nginx/ssl
    
    # Generate private key
    openssl genrsa -out nginx/ssl/key.pem 4096
    
    # Generate certificate signing request
    openssl req -new -key nginx/ssl/key.pem -out nginx/ssl/csr.pem -subj "/C=CA/ST=Ontario/L=Toronto/O=DDMV/CN=your-domain.com"
    
    # Generate self-signed certificate (replace with proper cert in production)
    openssl x509 -req -days 365 -in nginx/ssl/csr.pem -signkey nginx/ssl/key.pem -out nginx/ssl/cert.pem
    
    # Set proper permissions
    chmod 600 nginx/ssl/key.pem
    chmod 644 nginx/ssl/cert.pem
    
    log "SSL certificates generated"
}

# Setup firewall rules
setup_firewall() {
    log "Setting up firewall rules..."
    
    # Install ufw if not present
    if ! command -v ufw &> /dev/null; then
        apt-get update && apt-get install -y ufw
    fi
    
    # Reset firewall
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # SSH access
    ufw allow 22/tcp
    
    # HTTP/HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Monitoring (restrict to internal network)
    ufw allow from 10.0.0.0/8 to any port 3000  # Grafana
    ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus
    ufw allow from 10.0.0.0/8 to any port 5601  # Kibana
    
    # Database (internal only)
    ufw deny 5432/tcp
    ufw deny 1521/tcp
    ufw deny 6379/tcp
    
    # Enable firewall
    ufw --force enable
    
    log "Firewall configured"
}

# Harden system
harden_system() {
    log "Hardening system..."
    
    # Update system
    apt-get update && apt-get upgrade -y
    
    # Install security tools
    apt-get install -y \
        fail2ban \
        unattended-upgrades \
        logwatch \
        rkhunter \
        chkrootkit
    
    # Configure fail2ban
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
EOF
    
    # Configure automatic updates
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
    
    # Enable automatic updates
    echo 'APT::Periodic::Update-Package-Lists "1";' > /etc/apt/apt.conf.d/20auto-upgrades
    echo 'APT::Periodic::Unattended-Upgrade "1";' >> /etc/apt/apt.conf.d/20auto-upgrades
    
    # Start services
    systemctl enable fail2ban
    systemctl start fail2ban
    
    log "System hardened"
}

# Setup secrets management
setup_secrets() {
    log "Setting up secrets management..."
    
    # Create secrets directory
    mkdir -p /etc/ddmv/secrets
    chmod 700 /etc/ddmv/secrets
    
    # Generate secrets
    local db_password=$(generate_password)
    local oracle_password=$(generate_password)
    local redis_password=$(generate_password)
    local jwt_secret=$(generate_password)
    local secret_key=$(generate_password)
    local grafana_password=$(generate_password)
    
    # Create environment file
    cat > /etc/ddmv/secrets/.env.production << EOF
# Generated on $(date)
# DO NOT COMMIT TO VERSION CONTROL

# Database passwords
DB_PASSWORD=${db_password}
ORACLE_PASSWORD=${oracle_password}
REDIS_PASSWORD=${redis_password}

# Application secrets
SECRET_KEY=${secret_key}
JWT_SECRET=${jwt_secret}

# Monitoring
GRAFANA_PASSWORD=${grafana_password}

# External services (configure these manually)
SLACK_WEBHOOK_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
EMAIL_SMTP_PASSWORD=
EOF
    
    # Set proper permissions
    chmod 600 /etc/ddmv/secrets/.env.production
    chown root:root /etc/ddmv/secrets/.env.production
    
    log "Secrets generated and stored in /etc/ddmv/secrets/.env.production"
    warning "Please update external service credentials manually"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/ddmv << 'EOF'
/opt/ddmv-app/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose -f /opt/ddmv-app/docker-compose.prod.yml restart app nginx
    endscript
}

/var/log/nginx/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 nginx nginx
    postrotate
        docker-compose -f /opt/ddmv-app/docker-compose.prod.yml restart nginx
    endscript
}
EOF
    
    log "Log rotation configured"
}

# Setup monitoring for security events
setup_security_monitoring() {
    log "Setting up security monitoring..."
    
    # Install and configure auditd
    apt-get install -y auditd audispd-plugins
    
    # Configure audit rules
    cat > /etc/audit/rules.d/ddmv.rules << 'EOF'
# Monitor file access
-w /etc/ddmv/secrets/ -p wa -k secrets_access
-w /opt/ddmv-app/ -p wa -k app_files
-w /etc/nginx/ -p wa -k nginx_config
-w /etc/ssl/ -p wa -k ssl_certs

# Monitor authentication
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/group -p wa -k group_changes
-w /etc/sudoers -p wa -k sudoers_changes

# Monitor network connections
-a always,exit -F arch=b64 -S socket -k network_connect
-a always,exit -F arch=b32 -S socket -k network_connect

# Monitor privileged commands
-a always,exit -F path=/usr/bin/docker -F perm=x -F auid>=1000 -F auid!=4294967295 -k docker_exec
-a always,exit -F path=/usr/local/bin/docker-compose -F perm=x -F auid>=1000 -F auid!=4294967295 -k docker_compose_exec
EOF
    
    # Restart auditd
    service auditd restart
    
    log "Security monitoring configured"
}

# Setup intrusion detection
setup_intrusion_detection() {
    log "Setting up intrusion detection..."
    
    # Configure rkhunter
    rkhunter --update
    rkhunter --propupd
    
    # Schedule regular scans
    cat > /etc/cron.daily/security-scan << 'EOF'
#!/bin/bash
# Daily security scan

# Run rkhunter
/usr/bin/rkhunter --cronjob --update --quiet

# Run chkrootkit
/usr/sbin/chkrootkit | grep -v "nothing found" | grep -v "not infected" | mail -s "ChkRootkit Report" admin@your-domain.com

# Check for failed login attempts
grep "Failed password" /var/log/auth.log | tail -10 | mail -s "Failed Login Attempts" admin@your-domain.com
EOF
    
    chmod +x /etc/cron.daily/security-scan
    
    log "Intrusion detection configured"
}

# Backup security configuration
backup_security_config() {
    log "Creating security configuration backup..."
    
    local backup_dir="/opt/backups/security-$(date +%Y%m%d)"
    mkdir -p "$backup_dir"
    
    # Backup configurations
    cp -r /etc/ddmv "$backup_dir/"
    cp -r /etc/nginx "$backup_dir/"
    cp /etc/fail2ban/jail.local "$backup_dir/"
    cp /etc/audit/rules.d/ddmv.rules "$backup_dir/"
    
    # Create archive
    tar -czf "/opt/backups/security-config-$(date +%Y%m%d).tar.gz" -C "$backup_dir" .
    rm -rf "$backup_dir"
    
    log "Security configuration backed up"
}

# Main function
main() {
    log "Starting security setup for DDMV application..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
    
    # Run security setup
    generate_ssl_certs
    setup_firewall
    harden_system
    setup_secrets
    setup_log_rotation
    setup_security_monitoring
    setup_intrusion_detection
    backup_security_config
    
    log "Security setup completed successfully!"
    warning "Don't forget to:"
    warning "1. Update external service credentials in /etc/ddmv/secrets/.env.production"
    warning "2. Replace self-signed SSL certificates with proper ones"
    warning "3. Configure your domain name in nginx configuration"
    warning "4. Set up proper backup destinations"
    warning "5. Configure email notifications"
}

# Run main function
main "$@"
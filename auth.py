import streamlit as st
import hashlib
from datetime import datetime

class AdvancedAuthenticator:
    def __init__(self):
        self.users = {
            'admin': {
                'password': self.hash_password('admin123'),
                'role': 'Administrator',
                'name': 'System Admin',
                'email': 'admin@threatdetection.com'
            },
            'analyst': {
                'password': self.hash_password('analyst123'),
                'role': 'Security Analyst',
                'name': 'Security Analyst',
                'email': 'analyst@threatdetection.com'
            },
            'viewer': {
                'password': self.hash_password('viewer123'),
                'role': 'Viewer',
                'name': 'Report Viewer',
                'email': 'viewer@threatdetection.com'
            }
        }
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    def login(self, username, password):
        """Authenticate user"""
        # Check for breach attempts
        breach_attempts = self.detect_breach_attempts(username, password)
        
        if breach_attempts:
            st.warning(f"🚨 Security notice: {len(breach_attempts)} suspicious activities detected")
        
        if username in self.users and self.verify_password(password, self.users[username]['password']):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_role = self.users[username]['role']
            st.session_state.login_time = datetime.now()
            st.session_state.user_info = self.users[username]
            return True
        return False
    
    def logout(self):
        """Logout user"""
        for key in ['authenticated', 'username', 'user_role', 'login_time', 'user_info']:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_user_info(self):
        """Get current user information"""
        if st.session_state.get('authenticated') and st.session_state.get('username') in self.users:
            return self.users[st.session_state.username]
        return {
            'name': 'Guest',
            'role': 'Unknown',
            'email': 'guest@example.com'
        }
    
    def has_permission(self, required_role):
        """Check if user has required permission"""
        if not st.session_state.get('authenticated'):
            return False
            
        role_hierarchy = {
            'Viewer': 1,
            'Security Analyst': 2,
            'Administrator': 3
        }
        
        current_role = st.session_state.get('user_role')
        if current_role in role_hierarchy and required_role in role_hierarchy:
            return role_hierarchy[current_role] >= role_hierarchy[required_role]
        return False

    def detect_breach_attempts(self, username, password, ip_address=None):
        """Detect potential breach attempts during authentication"""
        breach_indicators = []
        
        # Initialize session state for login tracking
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = {}
        if 'suspicious_activities' not in st.session_state:
            st.session_state.suspicious_activities = []
        
        current_time = datetime.now()
        user_attempts = st.session_state.login_attempts.get(username, [])
        
        # Remove attempts older than 5 minutes
        user_attempts = [attempt for attempt in user_attempts 
                        if (current_time - attempt).seconds < 300]
        
        # Check if too many recent attempts (potential brute force)
        if len(user_attempts) >= 5:
            breach_indicators.append({
                'type': 'brute_force_attempt',
                'confidence': 0.9,
                'message': f"Multiple rapid login attempts for user {username}"
            })
            st.session_state.suspicious_activities.append({
                'timestamp': current_time,
                'type': 'brute_force',
                'username': username,
                'ip': ip_address
            })
        
        # Check for common breached passwords
        common_passwords = ['123456', 'password', 'admin', '12345678', 'qwerty']
        if password in common_passwords:
            breach_indicators.append({
                'type': 'weak_password',
                'confidence': 0.8,
                'message': "Common/weak password detected"
            })
        
        # Record this attempt
        user_attempts.append(current_time)
        st.session_state.login_attempts[username] = user_attempts
        
        return breach_indicators

    @property
    def authenticated(self):
        """Get authentication status"""
        return st.session_state.get('authenticated', False)

def setup_authentication():
    """Setup authentication system"""
    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = {}
    if 'suspicious_activities' not in st.session_state:
        st.session_state.suspicious_activities = []
    
    return AdvancedAuthenticator()

def show_login_form():
    """Display login form with dark text boxes"""
    st.markdown("""
    <style>
        /* Main background with animated gradient */
        .main .block-container {
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #667eea 0%, #764ba2 50%, #4A00E0 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            margin: 0;
            padding: 0;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Glass morphism login container */
        .login-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            padding: 50px 40px;
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                inset 0 -1px 0 rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            max-width: 480px;
            margin: 50px auto;
            color: white;
            text-align: center;
        }
        
        /* Login form styling */
        .login-form {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px 35px;
            border-radius: 20px;
            margin-top: 25px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* DARK TEXT BOXES - Enhanced styling */
        .stTextInput>div>div>input {
            background: rgba(45, 45, 65, 0.95) !important;
            border: 2px solid rgba(155, 89, 182, 0.4) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
            color: #ffffff !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        }
        
        .stTextInput>div>div>input::placeholder {
            color: rgba(255, 255, 255, 0.6) !important;
        }
        
        .stTextInput>div>div>input:focus {
            background: rgba(35, 35, 55, 0.95) !important;
            border-color: #9b59b6 !important;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.4),
                0 0 20px rgba(155, 89, 182, 0.6) !important;
            outline: none !important;
        }
        
        /* Enhanced login button */
        .stButton>button {
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 16px 32px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 25px rgba(155, 89, 182, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            width: 100% !important;
            margin-top: 20px !important;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #8e44ad 0%, #7d3c98 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 30px rgba(155, 89, 182, 0.6) !important;
        }
        
        /* Demo accounts section */
        .demo-accounts {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
            margin-top: 25px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: left;
        }
        
        /* Title styling */
        .main-title {
            font-size: 2.8em;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            font-weight: 300;
            margin-bottom: 30px;
        }
        
        /* Security icon animation */
        .security-icon {
            font-size: 4em;
            margin-bottom: 20px;
            display: inline-block;
            animation: pulse 2s infinite;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        /* Form group styling */
        .form-group {
            margin-bottom: 25px;
            text-align: left;
        }
        
        .form-label {
            color: #2c3e50;
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            display: block;
        }
        
        /* Remove default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Security alerts */
        .security-alert {
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.9) 0%, rgba(192, 57, 43, 0.9) 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin: 15px 0;
            border-left: 5px solid #c0392b;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="security-icon">🛡️</div>
            <h1 class="main-title">Cyber Threat Detection</h1>
            <p class="subtitle">Advanced ML-Powered Security Analytics Platform</p>
            
            <div class="login-form">
                <h3 style="color: #2c3e50; margin-bottom: 30px; font-size: 1.5em;">🔐 Secure Login</h3>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            # Username field with custom styling
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            st.markdown('<label class="form-label">👤 Username</label>', unsafe_allow_html=True)
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="username_input",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Password field with custom styling
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            st.markdown('<label class="form-label">🔒 Password</label>', unsafe_allow_html=True)
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="password_input",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            login_button = st.form_submit_button(
                "**🚀 Access Dashboard**", 
                use_container_width=True
            )
            
            if login_button:
                authenticator = st.session_state.get('authenticator')
                if authenticator and authenticator.login(username, password):
                    st.success(f"✅ Welcome {authenticator.get_user_info()['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Demo accounts info with enhanced styling
        st.markdown("""
        <div class="demo-accounts">
            <h4 style="color: white; margin-bottom: 15px; text-align: center; font-size: 1.1em;">🔑 Demo Access</h4>
            <div style="color: rgba(255,255,255,0.95); font-size: 0.95em;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>👨‍💼 <strong>Administrator</strong></span>
                    <span>admin / admin123</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span>🔍 <strong>Security Analyst</strong></span>
                    <span>analyst / analyst123</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>👁️ <strong>Viewer</strong></span>
                    <span>viewer / viewer123</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Security features
        st.markdown("""
        <div style="text-align: center; margin-top: 25px; color: rgba(255,255,255,0.8);">
            <h4 style="color: white; margin-bottom: 15px;">🛡️ Security Features</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85em;">
                <div>🔒 Brute Force Protection</div>
                <div>📊 Breach Detection</div>
                <div>🚨 Real-time Alerts</div>
                <div>🔍 Anomaly Detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add some space at the bottom
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
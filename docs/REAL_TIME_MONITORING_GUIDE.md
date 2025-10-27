# Real-Time Resource Monitoring System

## 🎯 **Overview**
We've implemented a comprehensive real-time monitoring system that gives users complete visibility into their resource usage, system status, and execution limits. This monitoring works for both **Custom Analysis** and **LLM-powered analysis**.

## 📊 **What Users Can See**

### **Quick Summary View (Always Visible)**
- **Executions per minute**: Current count vs. limit (e.g., 5/20)
- **Concurrent executions**: Active processes vs. limit (e.g., 1/2)
- **Status badge**: Color-coded status (Normal/Medium Usage/High Usage/Limited)
- **Progress bar**: Visual representation of usage levels

### **Detailed View (Expandable)**
#### **Usage Statistics**
- ✅ **Remaining Executions**: How many more code executions they can run
- ✅ **Active Processes**: How many code executions are currently running
- ✅ **Session Duration**: How long they've been active (MM:SS format)

#### **System Status**
- ✅ **Server Memory**: Real-time server memory usage percentage
- ✅ **Server CPU**: Real-time server CPU usage percentage  
- ✅ **Active Users**: How many users are currently using the system

#### **Resource Limits**
- ✅ **Memory Limit**: 512 MB per execution
- ✅ **CPU Limit**: 50% per execution
- ✅ **Timeout**: 30 seconds per execution

#### **Active Executions (When Running)**
- ✅ **Real-time tracking**: Shows which cells are executing
- ✅ **Execution duration**: How long each execution has been running
- ✅ **Visual indicators**: Animated spinners for active processes

## 🎨 **Visual Features**

### **Color-Coded System**
- 🟢 **Green**: Normal usage (0-60% of limits)
- 🟡 **Yellow**: Medium usage (60-80% of limits)
- 🔴 **Red**: High usage (80%+ of limits) or rate limited

### **Progress Bars**
- Dynamic width based on actual usage
- Color changes based on usage level
- Smooth animations for better user experience

### **Status Badges**
- "Normal" - All systems go
- "Medium Usage" - Approaching limits
- "High Usage" - Near limits
- "Rate Limited" - Temporarily blocked

## 🔄 **Real-Time Updates**

### **Auto-Refresh**
- Updates every **15 seconds** automatically
- Immediate updates after code execution
- No need for manual refresh

### **Event-Driven Updates**
- Updates when code execution starts
- Updates when code execution completes
- Updates when errors occur

## 🖥️ **Multi-Tab Support**

### **Custom Analysis Tab**
- **Full monitoring dashboard** with all features
- **Expandable detailed view** with system stats
- **Active execution tracking** with real-time duration

### **LLM Analysis Tab**
- **Simplified monitoring widget** 
- **Quick status overview** for AI-generated code
- **Same rate limits** apply to LLM executions

## 📱 **User Experience Benefits**

### **Transparency**
- Users know exactly how many executions they have left
- Clear visibility into system resource usage
- No surprises when hitting limits

### **Planning**
- Users can plan their analysis workflow
- Know when to expect rate limit resets
- Understand system load and performance

### **Feedback**
- Immediate feedback when approaching limits
- Clear error messages with retry timing
- Visual cues for system status

## 🔧 **Technical Implementation**

### **Backend APIs**
```python
GET /advanced-eda/api/rate-limit-status
# Returns user-specific rate limit information

GET /advanced-eda/api/system-stats  
# Returns detailed system resource statistics
```

### **Frontend Components**
```javascript
RateLimitMonitor.js
- Main monitoring widget
- Real-time updates
- Cross-tab synchronization
- Visual status indicators
```

### **Integration Points**
- Custom analysis code execution
- LLM-generated code execution
- Error handling and user feedback
- Session management

## 🎯 **Benefits for Users**

### **Better Resource Management**
- Users can see when they're approaching limits
- Plan complex analyses within their allocation
- Understand system capacity and performance

### **Improved Productivity**
- No unexpected execution failures
- Clear guidance on when to retry
- Optimal use of available resources

### **Enhanced Transparency**
- Full visibility into system status
- Understanding of fair usage policies
- Real-time feedback on resource consumption

### **Professional Experience**
- Enterprise-grade monitoring
- Clear, intuitive interface
- Consistent across all analysis methods

## 🚀 **Ready for Production**

This monitoring system provides:
- ✅ **Complete user visibility** into resource usage
- ✅ **Real-time updates** for responsive experience  
- ✅ **Multi-tab integration** for consistent experience
- ✅ **Professional UI/UX** with color-coded indicators
- ✅ **System transparency** for better user planning
- ✅ **Production-ready** monitoring and feedback

Users now have **complete control and visibility** over their resource usage, making the system transparent, fair, and professional for multi-user environments!

## 📸 **Visual Example**
```
🖥️ Resource Monitor                                    [▼]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Executions/Min    [████████░░] 8/20        Concurrent: 1/2    [Normal ✅]

▼ Details (Click to expand)
────────────────────────────────────────────────────────────
Usage Statistics              System Status
Remaining: 12                  Server Memory: 45.2% ✅
Active Processes: 1            Server CPU: 23.1% ✅  
Session: 5:23                  Active Users: 3

⚡ Active Executions
[🔄] Cell 3                                           12s
```

The system is now **fully production-ready** with comprehensive monitoring! 🎉
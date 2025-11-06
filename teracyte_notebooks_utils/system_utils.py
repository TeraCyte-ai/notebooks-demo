"""
System utilities for monitoring and managing system resources.
"""

import psutil
import time
import threading


def monitor_resources(threshold_ratio=0.85, critical_threshold=0.95, interval=5):
    """
    Monitors memory usage and issues visual alerts when usage exceeds thresholds.
    Prints alert only once per threshold crossing and auto-stops at critical level.
    
    Args:
        threshold_ratio (float): Fraction of total memory that triggers warning alert.
        critical_threshold (float): Fraction that triggers auto-stop to prevent OOM.
        interval (int): Sampling interval in seconds.
    """
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    threshold_gb = total_gb * threshold_ratio
    critical_gb = total_gb * critical_threshold
    
    # State tracking to print alerts only once
    warning_printed = False
    critical_printed = False

    def monitor():
        nonlocal warning_printed, critical_printed
        
        while True:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            used_percent = mem.percent
            
            # Critical threshold - auto-stop
            if used_gb > critical_gb:
                if not critical_printed:
                    print(
                        f"🚨 [CRITICAL] Memory usage critical - auto-stopping to prevent OOM\n"
                        f"   💾 Used: {used_gb:.2f} GB / {total_gb:.2f} GB ({used_percent:.1f}%)\n"
                        f"   🎯 Critical threshold: {critical_gb:.2f} GB ({critical_threshold*100:.0f}%)\n"
                        f"   🛑 Stopping execution to prevent OOM crash. Free some memory and try again."
                    )
                    critical_printed = True
                    # Stop execution
                    raise KeyboardInterrupt("Memory usage critical - auto-stopped to prevent OOM")
                    
            # Warning threshold
            elif used_gb > threshold_gb:
                if not warning_printed:
                    print(
                        f"⚠️  [WARNING] High memory usage detected\n"
                        f"   💾 Used: {used_gb:.2f} GB / {total_gb:.2f} GB ({used_percent:.1f}%)\n"
                        f"   🎯 Warning threshold: {threshold_gb:.2f} GB ({threshold_ratio*100:.0f}%)\n"
                        f"   🚨 Critical threshold: {critical_gb:.2f} GB ({critical_threshold*100:.0f}%)\n"
                    )
                    warning_printed = True
                    
            # Memory went back down - reset alerts
            else:
                if warning_printed or critical_printed:
                    print(f"✅ Memory usage back to normal: {used_percent:.1f}%")
                warning_printed = False
                critical_printed = False
            
            time.sleep(interval)

    print(f"🧠 Monitoring memory:")
    print(f"    - Warning at {threshold_ratio*100:.0f}% ({threshold_gb:.2f} GB)")
    print(f"    - Auto-stop at {critical_threshold*100:.0f}% ({critical_gb:.2f} GB)")
    threading.Thread(target=monitor, daemon=True).start()


def get_memory_info():
    """
    Get current memory usage information.
    
    Returns:
        dict: Dictionary containing memory usage statistics
    """
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024 ** 3),
        'used_gb': mem.used / (1024 ** 3),
        'available_gb': mem.available / (1024 ** 3),
        'percent_used': mem.percent
    }


def stop_if_memory_high(threshold=0.90):
    """
    Simple function: Stop execution if memory usage is too high.
    Call this before running memory-intensive operations.
    
    Args:
        threshold (float): Memory ratio that stops execution (default: 90%)
    """
    mem = psutil.virtual_memory()
    usage = mem.percent / 100.0
    
    if usage > threshold:
        print(f"🛑 STOPPING: Memory usage too high ({usage*100:.1f}% > {threshold*100:.0f}%)")
        print("   This prevents OOM crash. Free some memory and try again.")
        raise KeyboardInterrupt("Memory usage too high - stopping to prevent OOM")
    else:
        print(f"✅ Memory OK: {usage*100:.1f}% (< {threshold*100:.0f}%)")
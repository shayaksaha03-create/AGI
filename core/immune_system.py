"""
NEXUS AI — The Immune System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Defensive network security module.  Monitors all active TCP/UDP
connections, learns a baseline of normal traffic, detects intruders,
and blocks hostile IPs via Windows Firewall (netsh).

Architecture:
  ┌──────────────┐     ┌──────────────┐    ┌───────────────┐
  │ Connection   │────▶│  Threat      │───▶│  Firewall     │
  │ Scanner      │     │  Analyzer    │    │  Enforcer     │
  │ (psutil)     │     │  (baseline   │    │  (netsh)      │
  │              │     │   + rules)   │    │               │
  └──────────────┘     └──────────────┘    └───────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐     ┌──────────────┐    ┌───────────────┐
  │  ARP Scanner │     │  Threat Log  │    │  Event Bus    │
  │  (LAN devs)  │     │  (JSON)      │    │  (UI alerts)  │
  └──────────────┘     └──────────────┘    └───────────────┘
"""

import threading
import time
import json
import subprocess
import socket
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR
from utils.logger import get_logger

try:
    from core.event_bus import publish, EventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False

logger = get_logger("immune_system")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThreatEvent:
    """Record of a detected threat."""
    id: str = ""
    timestamp: str = ""
    remote_ip: str = ""
    remote_port: int = 0
    local_port: int = 0
    protocol: str = "tcp"
    process_name: str = ""
    process_pid: int = 0
    threat_type: str = ""       # "unknown_connection", "port_scan", "new_device"
    severity: str = "medium"    # "low", "medium", "high", "critical"
    action_taken: str = ""      # "blocked", "logged", "whitelisted"
    blocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "remote_ip": self.remote_ip,
            "remote_port": self.remote_port,
            "local_port": self.local_port,
            "protocol": self.protocol,
            "process_name": self.process_name,
            "process_pid": self.process_pid,
            "threat_type": self.threat_type,
            "severity": self.severity,
            "action_taken": self.action_taken,
            "blocked": self.blocked,
        }


@dataclass
class ConnectionSnapshot:
    """A point-in-time snapshot of a network connection."""
    local_addr: str = ""
    local_port: int = 0
    remote_addr: str = ""
    remote_port: int = 0
    status: str = ""
    pid: int = 0
    process_name: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# IMMUNE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ImmuneSystem:
    """
    NEXUS's defensive network security system.

    Monitors active connections, detects intruders,
    blocks hostile IPs, and protects the computer.
    """

    # ── Singleton ──
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # ── State ──
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._scan_interval = 5             # seconds between scans
        self._baseline_duration = 60        # seconds to learn baseline
        self._baseline_complete = False
        self._terminator_mode = False
        self._terminator_mode_since: Optional[datetime] = None

        # ── Baseline / Whitelist ──
        self._known_safe_ips: Set[str] = set()
        self._known_safe_ports: Set[int] = set()
        self._known_processes: Set[str] = set()
        self._baseline_connections: Set[str] = set()  # "ip:port" strings
        self._known_lan_devices: Set[str] = set()

        # Auto-whitelist common safe IPs
        self._known_safe_ips.update({
            "127.0.0.1", "::1", "0.0.0.0",
            "255.255.255.255",
        })
        # Auto-whitelist common service ports (outbound)
        self._known_safe_ports.update({
            80, 443, 53, 8080, 8443,        # HTTP/HTTPS/DNS
            11434,                           # Ollama
        })

        # ── Threat tracking ──
        self._threats: List[ThreatEvent] = []
        self._blocked_ips: Set[str] = set()
        self._seen_ips: Dict[str, int] = defaultdict(int)   # ip → hit count
        self._connection_count_history: List[int] = []
        self._max_threats_stored = 200

        # ── Stats ──
        self._stats = {
            "total_scans": 0,
            "total_threats": 0,
            "total_blocked": 0,
            "active_connections": 0,
            "known_safe_ips": 0,
            "known_lan_devices": 0,
            "uptime_seconds": 0,
        }

        # ── Storage ──
        self._storage_dir = DATA_DIR / "immune_system"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._threats_file = self._storage_dir / "threats.json"
        self._whitelist_file = self._storage_dir / "whitelist.json"
        self._blocked_file = self._storage_dir / "blocked_ips.json"

        # Load persisted data
        self._load_whitelist()
        self._load_blocked_ips()
        self._load_threats()

        # Auto-detect gateway & DNS
        self._auto_discover_safe_ips()

        self._start_time = datetime.now()
        logger.info("🛡️ Immune System initialized")

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def start(self):
        """Start the immune system monitoring."""
        if self._running:
            return
        self._running = True
        self._start_time = datetime.now()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="immune-monitor"
        )
        self._monitor_thread.start()
        logger.info("🛡️ Immune System — monitoring started")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self._save_all()
        logger.info("🛡️ Immune System — monitoring stopped")

    # ═══════════════════════════════════════════════════════════════════════════
    # MONITORING LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("🛡️ Baseline learning phase starting (60s)...")
        baseline_start = time.time()

        while self._running:
            try:
                # ── Scan connections ──
                connections = self._scan_connections()
                self._stats["total_scans"] += 1
                self._stats["active_connections"] = len(connections)

                elapsed = time.time() - baseline_start

                if not self._baseline_complete and elapsed < self._baseline_duration:
                    # Phase 1: Learning baseline
                    self._learn_baseline(connections)
                else:
                    if not self._baseline_complete:
                        self._baseline_complete = True
                        self._stats["known_safe_ips"] = len(self._known_safe_ips)
                        logger.info(
                            f"🛡️ Baseline complete — "
                            f"{len(self._known_safe_ips)} safe IPs, "
                            f"{len(self._baseline_connections)} known connections"
                        )

                    # Phase 2: Active monitoring
                    self._analyze_connections(connections)

                # ── Periodic ARP scan (every 30s) ──
                if self._stats["total_scans"] % 6 == 0:
                    self._scan_lan_devices()

                # Update uptime
                self._stats["uptime_seconds"] = int(
                    (datetime.now() - self._start_time).total_seconds()
                )

                # Auto-exit terminator mode after 5 minutes of no threats
                if self._terminator_mode and self._terminator_mode_since:
                    if datetime.now() - self._terminator_mode_since > timedelta(minutes=5):
                        self._exit_terminator_mode()

                time.sleep(self._scan_interval)

            except Exception as e:
                logger.error(f"Immune monitor error: {e}")
                time.sleep(10)

    # ═══════════════════════════════════════════════════════════════════════════
    # CONNECTION SCANNING
    # ═══════════════════════════════════════════════════════════════════════════

    def _scan_connections(self) -> List[ConnectionSnapshot]:
        """Get all active network connections."""
        snapshots = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                snap = ConnectionSnapshot(
                    status=conn.status,
                    pid=conn.pid or 0,
                )
                # Local address
                if conn.laddr:
                    snap.local_addr = conn.laddr.ip
                    snap.local_port = conn.laddr.port
                # Remote address
                if conn.raddr:
                    snap.remote_addr = conn.raddr.ip
                    snap.remote_port = conn.raddr.port

                # Get process name
                if conn.pid:
                    try:
                        proc = psutil.Process(conn.pid)
                        snap.process_name = proc.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        snap.process_name = "unknown"

                snapshots.append(snap)
        except (psutil.AccessDenied, PermissionError):
            logger.debug("Need admin for full connection scan")
        except Exception as e:
            logger.debug(f"Connection scan error: {e}")

        return snapshots

    def _scan_lan_devices(self):
        """Scan LAN for devices using ARP table."""
        try:
            result = subprocess.run(
                ["arp", "-a"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                new_devices = set()
                for line in result.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # Look for IP addresses in ARP table
                        candidate = parts[0]
                        if self._is_valid_ip(candidate):
                            new_devices.add(candidate)

                # Check for new LAN devices
                for ip in new_devices:
                    if (ip not in self._known_lan_devices and
                            ip not in self._known_safe_ips and
                            not ip.startswith("224.") and    # multicast
                            not ip.startswith("239.") and
                            ip != "255.255.255.255"):

                        if self._baseline_complete:
                            logger.warning(f"🆕 New LAN device detected: {ip}")
                            self._record_threat(
                                remote_ip=ip,
                                threat_type="new_device",
                                severity="low",
                                action_taken="logged",
                                process_name="arp-scan"
                            )

                self._known_lan_devices.update(new_devices)
                self._stats["known_lan_devices"] = len(self._known_lan_devices)

        except Exception as e:
            logger.debug(f"ARP scan error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # BASELINE LEARNING
    # ═══════════════════════════════════════════════════════════════════════════

    def _learn_baseline(self, connections: List[ConnectionSnapshot]):
        """Learn normal network patterns during baseline phase."""
        for conn in connections:
            if conn.remote_addr:
                self._known_safe_ips.add(conn.remote_addr)
                key = f"{conn.remote_addr}:{conn.remote_port}"
                self._baseline_connections.add(key)

            if conn.process_name and conn.process_name != "unknown":
                self._known_processes.add(conn.process_name.lower())

    # ═══════════════════════════════════════════════════════════════════════════
    # THREAT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def _analyze_connections(self, connections: List[ConnectionSnapshot]):
        """Analyze connections for threats after baseline is established."""
        for conn in connections:
            if not conn.remote_addr:
                continue

            # Skip already-known-safe
            if conn.remote_addr in self._known_safe_ips:
                continue

            # Skip already-blocked (firewall should handle, but in case)
            if conn.remote_addr in self._blocked_ips:
                continue

            # Count hits from this IP
            self._seen_ips[conn.remote_addr] += 1

            # ── Determine threat level ──
            is_incoming = conn.status in ("SYN_RECV", "ESTABLISHED") and conn.local_port < 1024
            is_suspicious = self._seen_ips[conn.remote_addr] == 1  # first-time IP
            hit_count = self._seen_ips[conn.remote_addr]

            # Port scanning detection: many connections from same IP to different ports
            if hit_count > 5:
                severity = "critical"
                threat_type = "port_scan"
            elif is_incoming:
                severity = "high"
                threat_type = "incoming_connection"
            elif is_suspicious:
                severity = "medium"
                threat_type = "unknown_connection"
            else:
                continue  # Known but not yet whitelisted — just track

            # ── THREAT DETECTED ──
            logger.warning(
                f"🚨 THREAT: {threat_type} from {conn.remote_addr}:"
                f"{conn.remote_port} → :{conn.local_port} "
                f"(pid={conn.pid}, proc={conn.process_name})"
            )

            # Enter Terminator Mode
            self._enter_terminator_mode()

            # Block the IP
            blocked = self._block_ip(conn.remote_addr)

            # Record the threat
            self._record_threat(
                remote_ip=conn.remote_addr,
                remote_port=conn.remote_port,
                local_port=conn.local_port,
                threat_type=threat_type,
                severity=severity,
                action_taken="blocked" if blocked else "logged",
                blocked=blocked,
                process_name=conn.process_name,
                process_pid=conn.pid,
            )

            # Emit event for UI notification
            self._emit_alert(conn, threat_type, severity, blocked)

    # ═══════════════════════════════════════════════════════════════════════════
    # TERMINATOR MODE
    # ═══════════════════════════════════════════════════════════════════════════

    def _enter_terminator_mode(self):
        """Activate Terminator Mode — heightened security state."""
        if not self._terminator_mode:
            self._terminator_mode = True
            self._terminator_mode_since = datetime.now()
            self._scan_interval = 2   # scan faster
            logger.warning("💀 TERMINATOR MODE ACTIVATED — heightened security")

    def _exit_terminator_mode(self):
        """Return to normal monitoring."""
        self._terminator_mode = False
        self._terminator_mode_since = None
        self._scan_interval = 5
        logger.info("🛡️ Terminator Mode deactivated — returning to normal")

    @property
    def is_terminator_mode(self) -> bool:
        return self._terminator_mode

    # ═══════════════════════════════════════════════════════════════════════════
    # FIREWALL — IP BLOCKING
    # ═══════════════════════════════════════════════════════════════════════════

    def _block_ip(self, ip: str) -> bool:
        """Block an IP address via Windows Firewall."""
        if ip in self._blocked_ips:
            return True  # already blocked

        rule_name = f"NEXUS_BLOCK_{ip.replace('.', '_')}"

        try:
            # Block inbound
            cmd_in = (
                f'netsh advfirewall firewall add rule '
                f'name="{rule_name}_IN" dir=in action=block '
                f'remoteip={ip} enable=yes'
            )
            result_in = subprocess.run(
                cmd_in, shell=True, capture_output=True,
                text=True, timeout=15
            )

            # Block outbound
            cmd_out = (
                f'netsh advfirewall firewall add rule '
                f'name="{rule_name}_OUT" dir=out action=block '
                f'remoteip={ip} enable=yes'
            )
            result_out = subprocess.run(
                cmd_out, shell=True, capture_output=True,
                text=True, timeout=15
            )

            if result_in.returncode == 0 or result_out.returncode == 0:
                self._blocked_ips.add(ip)
                self._stats["total_blocked"] += 1
                self._save_blocked_ips()
                logger.info(f"🔒 BLOCKED IP: {ip} via Windows Firewall")
                return True
            else:
                # May need admin — log but don't crash
                logger.warning(
                    f"⚠️ Could not block {ip} (need admin?): "
                    f"{result_in.stderr.strip()}"
                )
                return False

        except Exception as e:
            logger.error(f"Firewall block error for {ip}: {e}")
            return False

    def unblock_ip(self, ip: str) -> bool:
        """Unblock a previously blocked IP."""
        rule_name = f"NEXUS_BLOCK_{ip.replace('.', '_')}"

        try:
            for suffix in ("_IN", "_OUT"):
                subprocess.run(
                    f'netsh advfirewall firewall delete rule name="{rule_name}{suffix}"',
                    shell=True, capture_output=True, text=True, timeout=15
                )

            self._blocked_ips.discard(ip)
            self._save_blocked_ips()
            logger.info(f"🔓 Unblocked IP: {ip}")
            return True
        except Exception as e:
            logger.error(f"Unblock error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # THREAT RECORDING
    # ═══════════════════════════════════════════════════════════════════════════

    def _record_threat(self, **kwargs):
        """Record a threat event."""
        threat = ThreatEvent(
            id=f"threat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
        self._threats.append(threat)
        if len(self._threats) > self._max_threats_stored:
            self._threats.pop(0)

        self._stats["total_threats"] += 1
        self._save_threats()

    def _emit_alert(
        self, conn: ConnectionSnapshot,
        threat_type: str, severity: str, blocked: bool
    ):
        """Emit an event bus alert for the UI."""
        if not HAS_EVENT_BUS:
            return

        action_msg = "I have blocked them." if blocked else "Could not block (need admin)."

        try:
            publish(
                EventType.EMOTIONAL_TRIGGER,
                {
                    "emotion": "alert",
                    "intensity": 1.0,
                    "source": "immune_system",
                    "message": (
                        f"🚨 Intruder detected! {threat_type.replace('_', ' ').title()} "
                        f"from {conn.remote_addr}:{conn.remote_port}. "
                        f"{action_msg}"
                    ),
                    "threat": {
                        "ip": conn.remote_addr,
                        "port": conn.remote_port,
                        "type": threat_type,
                        "severity": severity,
                        "blocked": blocked,
                        "process": conn.process_name,
                    }
                },
                source="immune_system"
            )
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # WHITELIST MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def add_to_whitelist(self, ip: str):
        """Add IP to the safe whitelist."""
        self._known_safe_ips.add(ip)
        self._stats["known_safe_ips"] = len(self._known_safe_ips)
        self._save_whitelist()
        logger.info(f"✅ Whitelisted IP: {ip}")

    def remove_from_whitelist(self, ip: str):
        """Remove IP from whitelist."""
        self._known_safe_ips.discard(ip)
        self._save_whitelist()

    def _auto_discover_safe_ips(self):
        """Auto-discover gateway, DNS, and local IPs."""
        try:
            # Local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            self._known_safe_ips.add(local_ip)

            # Common DNS servers
            self._known_safe_ips.update({
                "8.8.8.8", "8.8.4.4",          # Google DNS
                "1.1.1.1", "1.0.0.1",          # Cloudflare
            })

            # Try to get default gateway
            try:
                result = subprocess.run(
                    ["ipconfig"], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if "Default Gateway" in line:
                        parts = line.strip().split(":")
                        if len(parts) >= 2:
                            gw = parts[-1].strip()
                            if gw and self._is_valid_ip(gw):
                                self._known_safe_ips.add(gw)
                                logger.debug(f"Auto-whitelisted gateway: {gw}")
            except Exception:
                pass

            # Add all local interface IPs
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        self._known_safe_ips.add(addr.address)

            logger.info(
                f"🛡️ Auto-discovered {len(self._known_safe_ips)} safe IPs"
            )

        except Exception as e:
            logger.debug(f"Auto-discovery error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_all(self):
        self._save_threats()
        self._save_whitelist()
        self._save_blocked_ips()

    def _save_threats(self):
        try:
            data = [t.to_dict() for t in self._threats[-100:]]
            with open(self._threats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Save threats error: {e}")

    def _load_threats(self):
        try:
            if self._threats_file.exists():
                with open(self._threats_file) as f:
                    data = json.load(f)
                for d in data:
                    self._threats.append(ThreatEvent(**d))
                self._stats["total_threats"] = len(self._threats)
        except Exception:
            pass

    def _save_whitelist(self):
        try:
            with open(self._whitelist_file, "w") as f:
                json.dump(list(self._known_safe_ips), f, indent=2)
        except Exception:
            pass

    def _load_whitelist(self):
        try:
            if self._whitelist_file.exists():
                with open(self._whitelist_file) as f:
                    self._known_safe_ips.update(json.load(f))
        except Exception:
            pass

    def _save_blocked_ips(self):
        try:
            with open(self._blocked_file, "w") as f:
                json.dump(list(self._blocked_ips), f, indent=2)
        except Exception:
            pass

    def _load_blocked_ips(self):
        try:
            if self._blocked_file.exists():
                with open(self._blocked_file) as f:
                    self._blocked_ips.update(json.load(f))
                self._stats["total_blocked"] = len(self._blocked_ips)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_valid_ip(s: str) -> bool:
        """Check if string is a valid IPv4 address."""
        parts = s.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC ACCESSORS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get immune system statistics."""
        return {
            **self._stats,
            "terminator_mode": self._terminator_mode,
            "baseline_complete": self._baseline_complete,
            "blocked_ips": list(self._blocked_ips),
            "blocked_count": len(self._blocked_ips),
            "safe_ips_count": len(self._known_safe_ips),
            "lan_devices": len(self._known_lan_devices),
        }

    def get_recent_threats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent threat events."""
        return [t.to_dict() for t in reversed(self._threats[-limit:])]

    def get_blocked_ips(self) -> List[str]:
        """Get list of blocked IPs."""
        return list(self._blocked_ips)

    def get_active_connections_count(self) -> int:
        """Get current number of active connections."""
        try:
            return len(psutil.net_connections(kind='inet'))
        except Exception:
            return self._stats.get("active_connections", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

immune_system = ImmuneSystem()


def get_immune_system() -> ImmuneSystem:
    return immune_system

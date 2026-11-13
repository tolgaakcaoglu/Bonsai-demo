#!/usr/bin/env python3
"""Lightweight XFCE/Tkinter manager for the local Bonsai llama.cpp server."""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

APP_DIR = Path.home() / "AI" / "Bonsai-demo"
START_SCRIPT = APP_DIR / "scripts" / "start_llama_server.sh"
CONFIG_FILE = Path.home() / ".config" / "bonsai-manager" / "config.json"
CUDA_LIBS = (
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda-13.4/targets/x86_64-linux/lib",
)


class BonsaiManager:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bonsai 2 — Yerel Sunucu Yöneticisi")
        self.root.geometry("900x690")
        self.root.minsize(760, 560)

        self.proc: subprocess.Popen[str] | None = None
        self.active_port: int | None = None
        self.active_context: int | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.restart_pending = False
        self.close_after_stop = False
        self.stop_started_at = 0.0
        self.force_sent = False
        self.last_status = ""

        # Donanım bilgileri çalışma anında okunur; hiçbir GPU/RAM değeri sabit kodlanmaz.
        self.system_info = self._detect_system_info()
        self.recommended_context = self._recommended_context()
        self.context_options = self._make_context_options()

        self.port_var = tk.StringVar(value=self._load_port())
        self.context_var = tk.StringVar(value=self._context_option(self._load_context()))
        self.system_info_var = tk.StringVar(value=self._system_info_text())
        self.status_var = tk.StringVar(value="Durum kontrol ediliyor…")
        self.url_var = tk.StringVar(value="")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_url()
        self._check_initial_port()
        self.root.after(100, self._drain_output)
        self.root.after(400, self._poll_process)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text="Bonsai 2 Sunucu Yöneticisi", font=("Sans", 16, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Ternary-Bonsai-2-27B • CUDA hızlandırma • Yerel API",
        ).pack(anchor="w", pady=(2, 14))

        port_frame = ttk.LabelFrame(outer, text="Sunucu bağlantısı", padding=10)
        port_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(port_frame, text="Port:").grid(row=0, column=0, sticky="w")
        self.port_entry = ttk.Entry(port_frame, textvariable=self.port_var, width=10)
        self.port_entry.grid(row=0, column=1, padx=(8, 8), sticky="w")
        ttk.Button(port_frame, text="Ayarları uygula", command=self._apply_port).grid(
            row=0, column=2, padx=(0, 16), sticky="w"
        )
        ttk.Label(port_frame, textvariable=self.url_var).grid(
            row=0, column=3, sticky="w"
        )
        port_frame.columnconfigure(3, weight=1)

        context_frame = ttk.LabelFrame(outer, text="Model bağlamı ve donanım tahmini", padding=10)
        context_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(context_frame, text="Context (token):").grid(row=0, column=0, sticky="w")
        self.context_combo = ttk.Combobox(
            context_frame,
            textvariable=self.context_var,
            values=self.context_options,
            state="readonly",
            width=34,
        )
        self.context_combo.grid(row=0, column=1, padx=(8, 10), sticky="w")
        ttk.Button(
            context_frame, text="Donanımı yenile", command=self._refresh_system_info
        ).grid(row=0, column=2, sticky="w")
        ttk.Label(
            context_frame,
            textvariable=self.system_info_var,
            wraplength=820,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(9, 0))
        context_frame.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(controls, text="▶ Başlat", command=self.start_server)
        self.start_button.pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(controls, text="■ Durdur", command=self.stop_server)
        self.stop_button.pack(side="left", padx=6)
        self.restart_button = ttk.Button(
            controls, text="↻ Yeniden başlat", command=self.restart_server
        )
        self.restart_button.pack(side="left", padx=6)
        ttk.Button(controls, text="Sohbeti aç", command=self.open_chat).pack(
            side="left", padx=6
        )

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(status_frame, text="Durum:", font=("Sans", 10, "bold")).pack(
            side="left"
        )
        ttk.Label(status_frame, textvariable=self.status_var).pack(
            side="left", padx=(6, 0)
        )

        logs_frame = ttk.LabelFrame(outer, text="Sunucu günlükleri", padding=6)
        logs_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            logs_frame,
            wrap="word",
            height=18,
            state="disabled",
            font=("Monospace", 9),
            padx=8,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(
            outer,
            text="API adresi: http://127.0.0.1:<port>/v1  •  Varsayılan bağlanma adresi yalnızca bu bilgisayardır.",
        ).pack(anchor="w", pady=(8, 0))

    @staticmethod
    def _detect_system_info() -> dict[str, object]:
        """Read total/available system RAM and NVIDIA GPU VRAM when available."""
        info: dict[str, object] = {
            "ram_total_gib": None,
            "ram_available_gib": None,
            "gpu_name": None,
            "vram_total_gib": None,
            "vram_free_gib": None,
        }
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as mem_file:
                for line in mem_file:
                    match = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)", line)
                    if match:
                        meminfo[match.group(1)] = int(match.group(2))
            if "MemTotal" in meminfo:
                info["ram_total_gib"] = meminfo["MemTotal"] / (1024 * 1024)
            if "MemAvailable" in meminfo:
                info["ram_available_gib"] = meminfo["MemAvailable"] / (1024 * 1024)
        except OSError:
            pass

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                fields = next(csv.reader([result.stdout.strip().splitlines()[0]]))
                if len(fields) >= 3:
                    info["gpu_name"] = fields[0].strip()
                    # nvidia-smi reports memory in MiB with nounits.
                    info["vram_total_gib"] = float(fields[1].strip()) / 1024
                    info["vram_free_gib"] = float(fields[2].strip()) / 1024
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
            pass
        return info

    def _recommended_context(self) -> int:
        """Conservative heuristic for this bundled 27B PQ2_0 model, not a benchmark."""
        vram = self.system_info.get("vram_total_gib")
        ram = self.system_info.get("ram_total_gib")
        if not isinstance(vram, (int, float)) or not isinstance(ram, (int, float)):
            return 4096
        if vram < 8 or ram < 12:
            return 4096
        if vram < 12 or ram < 24:
            return 8192
        if vram < 16 or ram < 32:
            return 12288
        if vram < 24 or ram < 48:
            return 16384
        if vram < 32 or ram < 64:
            return 24576
        return 32768

    def _context_estimate(self, context: int) -> str:
        # Estimate relative to a conservative recommended context computed from detected hardware.
        recommended = self.recommended_context
        if context <= recommended / 2:
            return "çok hızlı"
        if context <= recommended:
            return "hızlı · önerilen"
        if context <= recommended * 2:
            return "normal"
        if context <= recommended * 3:
            return "yavaş · bellek yükü"
        return "çok yavaş · bellek riski"

    def _context_option(self, value: object) -> str:
        try:
            context = int(str(value).split()[0])
        except (TypeError, ValueError):
            context = 8192
        choices = (4096, 8192, 12288, 16384, 20480, 24576, 32768)
        if context not in choices:
            context = 8192
        return f"{context} — {self._context_estimate(context)}"

    def _make_context_options(self) -> tuple[str, ...]:
        return tuple(
            f"{value} — {self._context_estimate(value)}"
            for value in (4096, 8192, 12288, 16384, 20480, 24576, 32768)
        )

    def _system_info_text(self) -> str:
        ram = self.system_info.get("ram_total_gib")
        ram_free = self.system_info.get("ram_available_gib")
        vram = self.system_info.get("vram_total_gib")
        vram_free = self.system_info.get("vram_free_gib")
        gpu_name = self.system_info.get("gpu_name")
        parts: list[str] = ["Algılanan donanım:"]
        if gpu_name and isinstance(vram, (int, float)):
            gpu_text = f"{gpu_name} · VRAM {vram:.1f} GiB"
            if isinstance(vram_free, (int, float)):
                gpu_text += f" ({vram_free:.1f} GiB boş)"
            parts.append(gpu_text)
        else:
            parts.append("NVIDIA GPU/VRAM bilgisi okunamadı")
        if isinstance(ram, (int, float)):
            ram_text = f"RAM {ram:.1f} GiB"
            if isinstance(ram_free, (int, float)):
                ram_text += f" ({ram_free:.1f} GiB kullanılabilir)"
            parts.append(ram_text)
        else:
            parts.append("RAM bilgisi okunamadı")
        parts.append(f"Bu donanım için kaba başlangıç önerisi: {self.recommended_context} token.")
        parts.append(
            "Hız etiketleri tahminidir; gerçek hız kullanılan bağlam uzunluğuna, boş belleğe ve GPU katmanlarının yüklenmesine göre değişir."
        )
        if isinstance(ram_free, (int, float)) and ram_free < 4:
            parts.append("Uyarı: kullanılabilir RAM 4 GiB altında; yüksek context ayarlarından kaçının.")
        if isinstance(vram_free, (int, float)) and vram_free < 2:
            parts.append("Uyarı: kullanılabilir VRAM 2 GiB altında; model başlatılırken bellek hatası yaşanabilir.")
        return "  •  ".join(parts)

    def _refresh_system_info(self) -> None:
        current_context = self._get_context(show_error=False) or 8192
        self.system_info = self._detect_system_info()
        self.recommended_context = self._recommended_context()
        self.context_options = self._make_context_options()
        self.context_combo.configure(values=self.context_options)
        self.context_var.set(self._context_option(current_context))
        self.system_info_var.set(self._system_info_text())
        self._append_log("Donanım bilgileri yeniden okundu; context tahminleri güncellendi.")

    def _load_port(self) -> str:
        try:
            value = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("port", 8081)
            return str(value)
        except (OSError, ValueError, AttributeError):
            return "8081"

    def _load_context(self) -> str:
        try:
            value = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("context", 16384)
            return str(value)
        except (OSError, ValueError, AttributeError):
            return "16384"

    def _save_settings(self, port: int, context: int) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps({"port": port, "context": context}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _get_port(self, show_error: bool = True) -> int | None:
        raw = self.port_var.get().strip()
        try:
            port = int(raw)
            if not 1024 <= port <= 65535:
                raise ValueError
        except ValueError:
            if show_error:
                messagebox.showerror("Geçersiz port", "1024 ile 65535 arasında bir port numarası gir.", parent=self.root)
            return None
        return port

    def _get_context(self, show_error: bool = True) -> int | None:
        raw = self.context_var.get().strip()
        try:
            match = re.match(r"^(\d+)", raw)
            if match is None:
                raise ValueError
            context = int(match.group(1))
            if context not in (4096, 8192, 12288, 16384, 20480, 24576, 32768):
                raise ValueError
        except ValueError:
            if show_error:
                messagebox.showerror(
                    "Geçersiz context",
                    "Açılır listeden context uzunluğu seç.",
                    parent=self.root,
                )
            return None
        return context

    def _update_url(self) -> None:
        port = self._get_port(show_error=False)
        self.url_var.set(f"http://localhost:{port}" if port else "Geçersiz port")

    @staticmethod
    def _port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _health_ok(port: int) -> bool:
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(request, timeout=0.35) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def _check_initial_port(self) -> None:
        port = self._get_port(show_error=False)
        other_ports = [p for p in (8080, 8081) if p != port and self._health_ok(p)]
        if other_ports:
            self.status_var.set(f"Mevcut Bonsai sunucusu {other_ports[0]} portunda algılandı; ikinci örnek başlatılmasın.")
            self._append_log(f"Uyarı: Bonsai sunucusu {other_ports[0]} portunda zaten çalışıyor. Aynı anda iki 27B model açmamak için önce eski terminalde Ctrl+C ile durdur.")
        elif not port:
            self.status_var.set("Port numarası geçersiz.")
        elif self._port_open(port):
            if self._health_ok(port):
                self.status_var.set(f"Port {port} üzerinde bir Bonsai/API sunucusu zaten çalışıyor; bu pencere başlatmadı.")
            else:
                self.status_var.set(f"Port {port} kullanımda; başka bir uygulama dinliyor olabilir.")
            self._append_log("Not: Başka bir terminalden başlatılmış sunucuyu bu pencere yönetemez. İlk kullanımda eski sunucuyu Ctrl+C ile durdurup buradan başlat.")
        else:
            self.status_var.set("Durduruldu — başlatılmaya hazır.")
        self._update_buttons()

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        try:
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > 1200:
                self.log_text.delete("1.0", f"{line_count - 1000}.0")
        except (ValueError, tk.TclError):
            pass
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_buttons(self) -> None:
        running = self.proc is not None and self.proc.poll() is None
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.restart_button.configure(state="normal")

    def _apply_port(self) -> None:
        port = self._get_port()
        if port is None:
            return
        context = self._get_context()
        if context is None:
            return
        old_port = self.active_port
        old_context = self.active_context
        self._save_settings(port, context)
        self._update_url()
        if self.proc is not None and self.proc.poll() is None and (old_port != port or old_context != context):
            if messagebox.askyesno(
                "Ayar değişikliği",
                f"Sunucu port {old_port}, context {old_context} ile çalışıyor.\n\n"
                f"Port {port} ve context {context} ile yeniden başlatılsın mı?",
                parent=self.root,
            ):
                self.restart_pending = True
                self.stop_server()
                return
            self.status_var.set(
                f"Mevcut sunucu port {old_port} / context {old_context} ile çalışıyor; yeni ayarlar sonraki başlatmada kullanılacak."
            )
        else:
            self.status_var.set(f"Ayarlar kaydedildi — port {port}, context {context}.")
        self._append_log(f"Ayarlar kaydedildi: port={port}, context={context}")

    def start_server(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.status_var.set(f"Sunucu zaten çalışıyor (port {self.active_port}).")
            return
        port = self._get_port()
        if port is None:
            return
        context = self._get_context()
        if context is None:
            return
        if not START_SCRIPT.is_file():
            messagebox.showerror("Dosya bulunamadı", f"Başlatma betiği bulunamadı:\n{START_SCRIPT}", parent=self.root)
            return
        other_ports = [p for p in (8080, 8081) if p != port and self._health_ok(p)]
        if other_ports:
            existing_port = other_ports[0]
            messagebox.showerror(
                "Bonsai zaten çalışıyor",
                f"Bonsai sunucusu {existing_port} portunda zaten yanıt veriyor.\n\nİki 27B modeli aynı anda başlatmak VRAM'i tüketebilir. Önce eski terminalde Ctrl+C ile çalışan sunucuyu durdur, sonra buradan Başlat'a bas.",
                parent=self.root,
            )
            self.status_var.set(f"Mevcut Bonsai {existing_port} portunda çalışıyor; ikinci örnek başlatılmadı.")
            return
        if self._port_open(port):
            messagebox.showerror(
                "Port kullanımda",
                f"Port {port} zaten kullanımda.\n\nBonsai daha önce bir terminalden başlatıldıysa o terminale dönüp Ctrl+C ile durdur. Başka bir uygulama bu portu kullanıyorsa farklı bir port seç.",
                parent=self.root,
            )
            self.status_var.set(f"Port {port} kullanımda; yeni sunucu başlatılmadı.")
            return

        env = os.environ.copy()
        old_ld = env.get("LD_LIBRARY_PATH", "")
        ld_parts = [*CUDA_LIBS, *([old_ld] if old_ld else [])]
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(ld_parts))
        env["PORT"] = str(port)
        env["BONSAI_CTX"] = str(context)
        env["BONSAI_MMPROJ_CPU"] = "1"
        env["BONSAI_HOST"] = "127.0.0.1"

        try:
            self._save_settings(port, context)
            self.proc = subprocess.Popen(
                [str(START_SCRIPT)],
                cwd=str(APP_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.proc = None
            messagebox.showerror("Başlatma hatası", str(exc), parent=self.root)
            self.status_var.set("Başlatılamadı.")
            self._update_buttons()
            return

        self.active_port = port
        self.active_context = context
        self.status_var.set(f"Başlatılıyor — port {port}, context {context}…")
        self._append_log(f"\n=== Bonsai başlatılıyor | port {port} | context {context} ===")
        self._append_log(f"Sohbet: http://localhost:{port} | API: http://127.0.0.1:{port}/v1")
        threading.Thread(target=self._read_process_output, args=(self.proc,), daemon=True).start()
        self._update_buttons()

    def _read_process_output(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                self.output_queue.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass

    def _drain_output(self) -> None:
        try:
            while True:
                self._append_log(self.output_queue.get_nowait())
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self._drain_output)

    def stop_server(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.status_var.set("Bu pencerenin yönettiği çalışan bir sunucu yok.")
            self._update_buttons()
            return
        self.restart_pending = self.restart_pending or False
        self.stop_started_at = time.monotonic()
        self.force_sent = False
        self.status_var.set("Sunucu durduruluyor…")
        self._append_log("Sunucuyu durdurma sinyali gönderildi.")
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._wait_for_stop()

    def restart_server(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.restart_pending = True
            self.stop_server()
        else:
            self.start_server()

    def _wait_for_stop(self) -> None:
        if self.proc is None:
            if self.close_after_stop:
                self.root.destroy()
            return
        code = self.proc.poll()
        if code is not None:
            self.proc = None
            self.active_port = None
            self.active_context = None
            self._append_log(f"Sunucu durdu (çıkış kodu {code}).")
            self._update_buttons()
            if self.close_after_stop:
                self.root.destroy()
                return
            if self.restart_pending:
                self.restart_pending = False
                self.root.after(350, self.start_server)
            else:
                self.status_var.set("Durduruldu.")
            return
        if time.monotonic() - self.stop_started_at > 8 and not self.force_sent:
            self.force_sent = True
            self._append_log("Sunucu kapanmadı; zorla durdurma sinyali gönderiliyor.")
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.root.after(250, self._wait_for_stop)

    def _poll_process(self) -> None:
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.proc = None
                self.active_port = None
                self.active_context = None
                self._update_buttons()
                if self.close_after_stop:
                    self.root.destroy()
                    return
                if self.restart_pending:
                    self.restart_pending = False
                    self.root.after(350, self.start_server)
                else:
                    self.status_var.set(f"Sunucu durdu (çıkış kodu {code}).")
            elif self.active_port and self._health_ok(self.active_port):
                self.status_var.set(f"Çalışıyor — http://localhost:{self.active_port}")
            elif self.active_port:
                self.status_var.set(f"Model yükleniyor — port {self.active_port}…")
            self._update_buttons()
        if self.root.winfo_exists():
            self.root.after(700, self._poll_process)

    def open_chat(self) -> None:
        port = self.active_port or self._get_port()
        if port is None:
            return
        webbrowser.open(f"http://localhost:{port}")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            should_stop = messagebox.askyesno(
                "Bonsai çalışıyor",
                "Uygulamadan çıkarken Bonsai sunucusu da durdurulsun mu?\n\nEvet: sunucuyu durdur ve çık.\nHayır: pencere açık kalsın.",
                parent=self.root,
            )
            if should_stop:
                self.close_after_stop = True
                self.restart_pending = False
                self.stop_server()
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    BonsaiManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()

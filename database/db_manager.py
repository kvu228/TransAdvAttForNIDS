"""
Database manager for SPTS NIDS Simulation - SQLite3 Data Layer (Tier 3).

Quản lý lưu trữ lịch sử mô phỏng tấn công. Sử dụng SQLite3 (không cần server riêng).
File database: simulation_logs.db trong project root.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

# Project root: parent of database/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_ROOT, "simulation_logs.db")


def _get_conn() -> sqlite3.Connection:
    """Tạo kết nối SQLite tới simulation_logs.db."""
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    """
    Khởi tạo database và bảng Attack_Simulations nếu chưa tồn tại.

    Bảng gồm: id, timestamp, algorithm, iterations, step_size,
    target_model, surrogate_model, evasion_rate, copies, dropout_rate.
    """
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Attack_Simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            iterations INTEGER NOT NULL,
            step_size REAL NOT NULL,
            target_model TEXT NOT NULL,
            surrogate_model TEXT NOT NULL,
            evasion_rate REAL NOT NULL,
            copies INTEGER,
            dropout_rate REAL
        )
    """)
    conn.commit()
    conn.close()


def log_simulation(
    timestamp: str,
    algorithm: str,
    iterations: int,
    step_size: float,
    target_model: str,
    surrogate_model: str,
    evasion_rate: float,
    copies: Optional[int] = None,
    dropout_rate: Optional[float] = None,
) -> None:
    """
    Ghi một bản ghi mô phỏng vào database.

    Args:
        timestamp: Thời gian chạy (ISO format).
        algorithm: MIFGSM | SIM | VMIFGSM | DGM.
        iterations, step_size: Tham số tấn công.
        target_model, surrogate_model: Tên model.
        evasion_rate: Tỷ lệ bypass NIDS (%).
        copies, dropout_rate: Tham số DGM (None nếu không dùng DGM).
    """
    init_db()
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO Attack_Simulations
        (timestamp, algorithm, iterations, step_size, target_model, surrogate_model, evasion_rate, copies, dropout_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, algorithm, iterations, step_size, target_model, surrogate_model, evasion_rate, copies, dropout_rate),
    )
    conn.commit()
    conn.close()


def get_history():
    """
    Lấy toàn bộ lịch sử mô phỏng, sắp xếp theo thời gian mới nhất trước.

    Returns:
        List[dict]: Mỗi dict là một bản ghi với keys: id, timestamp, algorithm, ...
    """
    init_db()
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, algorithm, iterations, step_size,
               target_model, surrogate_model, evasion_rate, copies, dropout_rate
        FROM Attack_Simulations
        ORDER BY timestamp DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

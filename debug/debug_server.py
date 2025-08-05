"""
Debug utilities for CrystalNexus development
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import app
import uvicorn

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug/debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_debug_server():
    """Run the FastAPI server in debug mode"""
    logger.info("Starting CrystalNexus in debug mode")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="debug"
    )

if __name__ == "__main__":
    run_debug_server()
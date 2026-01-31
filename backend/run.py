#!/usr/bin/env python3
"""
Run the DataChat Analytics Platform backend server.
"""

import uvicorn
import os

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting DataChat Analytics Platform on {host}:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

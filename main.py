"""
FastAPI Backend for iPhone Demand-Supply Chatbot
Uses Gemini + Optimizer.json for SQL Generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import duckdb
import json
import os
from typing import Optional, Dict, Any, List
import re

# ==========================================
# Configuration
# ==========================================

app = FastAPI(
    title="iPhone Demand-Supply Chatbot API",
    description="Natural language to SQL chatbot for iPhone inventory management",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน production ควรระบุ domain จริง
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Global Variables
# ==========================================

# Database connection
con = None
optimizer = None
gemini_model = None

# ==========================================
# Models
# ==========================================

class KeyMetric(BaseModel):
    label: str
    value: str
    unit: str

class StructuredAnalysis(BaseModel):
    summary: str
    key_metrics: List[KeyMetric]
    details: Optional[List[str]] = None
    insight: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    include_sql: Optional[bool] = True
    include_data: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    structured: Optional[Dict[str, Any]] = None

# ==========================================
# Startup & Shutdown
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Load resources on startup"""
    global con, optimizer, gemini_model
    
    print("🚀 Starting FastAPI server...")
    
    # 1. Load optimizer
    print("📦 Loading optimizer.json...")
    if os.path.exists('optimizer.json'):
        with open('optimizer.json', 'r', encoding='utf-8') as f:
            optimizer = json.load(f)
        print(f"✅ Optimizer loaded: {optimizer['metadata']['total_examples']} examples")
    else:
        raise FileNotFoundError("optimizer.json not found! Please run Colab script first.")
    
    # 2. Setup database (in-memory for cloud deployment)
    print("💾 Setting up in-memory database...")
    con = duckdb.connect(':memory:')  # Use in-memory database
    
    # Load CSV files into memory
    print("📁 Loading data from CSVs...")
    try:
        con.execute("CREATE TABLE dim_date AS SELECT * FROM 'dim_date.csv'")
        con.execute("CREATE TABLE dim_branch AS SELECT * FROM 'dim_branch.csv'")
        con.execute("CREATE TABLE dim_product AS SELECT * FROM 'dim_product.csv'")
        con.execute("CREATE TABLE fact_registration AS SELECT * FROM 'fact_registration.csv'")
        con.execute("CREATE TABLE fact_contract AS SELECT * FROM 'fact_contract.csv'")
        con.execute("CREATE TABLE fact_inventory AS SELECT * FROM 'fact_inventory.csv'")
        print("✅ Database loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading CSVs: {e}")
        raise
    
    # 3. Configure Gemini
    print("🤖 Configuring Gemini...")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables")
        print("   Please set it: export GEMINI_API_KEY=your_key")
    else:
        genai.configure(api_key=gemini_api_key)
        # Use Gemini 2.5 Flash - latest model with best performance
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini 2.5 Flash configured")
    
    print("✅ Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global con
    if con:
        con.close()
        print("💾 Database connection closed")

# ==========================================
# Helper Functions
# ==========================================

def clean_sql(sql_text: str) -> str:
    """Clean SQL from markdown and extra text"""
    # Remove markdown code blocks
    sql_text = re.sub(r'```sql\n?', '', sql_text)
    sql_text = re.sub(r'```\n?', '', sql_text)
    
    # Remove common prefixes
    sql_text = re.sub(r'^(SQL:|Query:)\s*', '', sql_text, flags=re.IGNORECASE)
    
    # Strip whitespace
    sql_text = sql_text.strip()
    
    return sql_text

def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate SQL query
    Returns: (is_valid, error_message)
    """
    # Basic validation
    sql_upper = sql.upper()
    
    # Check for dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous operation detected: {keyword}"
    
    # Must be SELECT query
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    # Check for valid table names
    valid_tables = ['dim_date', 'dim_branch', 'dim_product', 
                   'fact_registration', 'fact_contract', 'fact_inventory']
    
    # Simple check - look for FROM/JOIN clauses
    has_valid_table = any(table in sql for table in valid_tables)
    if not has_valid_table:
        return False, "No valid table found in query"
    
    return True, ""

def generate_sql_with_gemini(question: str, optimizer: dict) -> str:
    """
    Generate SQL using Gemini with optimizer context
    """
    # Get current date (for default time filters)
    current_date = con.execute("SELECT MAX(date_key) FROM fact_inventory").fetchone()[0]
    
    # Create prompt
    prompt = f"""You are an expert SQL query generator for a DuckDB database.

DATABASE SCHEMA:
{json.dumps(optimizer['schema']['tables'], indent=2, ensure_ascii=False)}

BUSINESS RULES:
{json.dumps(optimizer['business_glossary'], indent=2, ensure_ascii=False)}

DEFAULT ASSUMPTIONS:
{json.dumps(optimizer['default_assumptions'], indent=2, ensure_ascii=False)}

EXAMPLE QUESTIONS AND SQL (Learn from these patterns):
{json.dumps(optimizer['examples'][:10], indent=2, ensure_ascii=False)}

IMPORTANT NOTES:
- Current latest date in database: {current_date}
- Use date_key format: YYYYMMDD (e.g., 20251227)
- For "today" or "วันนี้", use: WHERE date_key = {current_date}
- For "this month" or "เดือนนี้", use: WHERE date_key BETWEEN {str(current_date)[:6]}01 AND {str(current_date)[:6]}31
- For inventory queries, use latest date: WHERE date_key = (SELECT MAX(date_key) FROM fact_inventory)
- Only shops (BR001-BR005) have registrations and contracts
- Warehouse (BR000) only appears in inventory

USER QUESTION (in Thai):
{question}

Generate ONLY the SQL query. No explanation, no markdown, just the SQL.
The SQL must be valid DuckDB syntax.
"""

    try:
        response = gemini_model.generate_content(prompt)
        sql = clean_sql(response.text)
        return sql
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

def format_response_with_gemini(question: str, data: List[Dict], sql: str) -> tuple[str, Optional[Dict]]:
    """
    Format data into structured markdown response using Gemini
    Returns: (formatted_markdown, structured_json)
    """
    
    # Prompt for structured JSON output
    prompt = f"""Analyze this business data and respond in JSON format ONLY.

USER QUESTION: {question}

SQL EXECUTED: {sql}

QUERY RESULTS:
{json.dumps(data, indent=2, ensure_ascii=False)}

Return a valid JSON object with this EXACT structure:
{{
    "summary": "คำตอบสั้นๆ 1-2 ประโยค พร้อมตัวเลขสำคัญที่สุด",
    "key_metrics": [
        {{"label": "ชื่อตัวชี้วัด", "value": "ตัวเลข", "unit": "หน่วย"}},
        {{"label": "ชื่อตัวชี้วัด", "value": "ตัวเลข", "unit": "หน่วย"}}
    ],
    "details": [
        "รายละเอียดสำคัญข้อ 1 (สั้นๆ ไม่เกิน 1 บรรทัด)",
        "รายละเอียดสำคัญข้อ 2 (สั้นๆ ไม่เกิน 1 บรรทัด)"
    ],
    "insight": "ข้อสังเกตหรือคำแนะนำสำคัญ 1 ประโยค"
}}

RULES:
- summary: ตอบคำถามตรงประเด็น ไม่เกิน 2 ประโยค
- key_metrics: 2-5 ตัวเลขสำคัญที่สุด (เรียงตามความสำคัญ)
- details: 2-4 ข้อ แต่ละข้อสั้นๆ ไม่ซ้ำกับ summary
- insight: ข้อสังเกต/แนวโน้ม/คำแนะนำ 1 ประโยค (ถ้าไม่มี ใส่ null)
- ใช้ตัวเลขที่มี comma (เช่น 10,032 ไม่ใช่ 10032)
- เขียนเป็นภาษาไทยที่เป็นทางการแต่เข้าใจง่าย
- Return ONLY valid JSON, no markdown backticks, no extra text

Example response:
{{
    "summary": "iPhone 17 มีลูกค้าลงทะเบียนรอทำสัญญาทั้งหมด 10,032 คน",
    "key_metrics": [
        {{"label": "iPhone 17 256GB Blue", "value": "10,032", "unit": "คน"}},
        {{"label": "Stock พร้อมส่ง", "value": "152,985", "unit": "เครื่อง"}},
        {{"label": "อัตราส่วน Stock/Demand", "value": "15.2", "unit": "เท่า"}}
    ],
    "details": [
        "Stock มากกว่า Demand ประมาณ 15 เท่า",
        "ไม่มีความเสี่ยงในการขาดแคลนสินค้า"
    ],
    "insight": "สามารถรองรับ Demand ได้เป็นอย่างดี แนะนำติดตาม Conversion Rate"
}}

Now analyze the data and return JSON:"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean potential markdown formatting
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        structured_data = json.loads(response_text)
        
        # Validate structure
        if not all(key in structured_data for key in ['summary', 'key_metrics']):
            raise ValueError("Missing required keys")
        
        # Format as beautiful markdown
        markdown_parts = []
        
        # Summary with emoji
        markdown_parts.append(f"**📊 {structured_data['summary']}**")
        
        # Key Metrics
        if structured_data.get('key_metrics'):
            markdown_parts.append("\n**📈 ตัวเลขสำคัญ:**")
            for metric in structured_data['key_metrics']:
                markdown_parts.append(f"• **{metric['label']}:** {metric['value']} {metric['unit']}")
        
        # Details
        if structured_data.get('details'):
            markdown_parts.append("\n**📋 รายละเอียด:**")
            for detail in structured_data['details']:
                markdown_parts.append(f"• {detail}")
        
        # Insight
        if structured_data.get('insight'):
            markdown_parts.append(f"\n**💡 Insight:** {structured_data['insight']}")
        
        formatted_markdown = '\n'.join(markdown_parts)
        
        return formatted_markdown, structured_data
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Parse Error: {e}")
        print(f"Response: {response_text[:200]}...")
        # Fallback to simple formatting
        fallback = f"**📊 สรุป:** พบข้อมูล {len(data)} รายการ\n\n{json.dumps(data[:3], indent=2, ensure_ascii=False)}"
        return fallback, None
        
    except Exception as e:
        print(f"⚠️ Formatting Error: {e}")
        # Fallback
        fallback = f"**ผลลัพธ์:** พบข้อมูล {len(data)} รายการ"
        return fallback, None

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "iPhone Demand-Supply Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Submit natural language query",
            "GET /schema": "Get database schema",
            "GET /examples": "Get example questions",
            "GET /stats": "Get database statistics"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main endpoint: Convert natural language to SQL and execute
    """
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # 1. Generate SQL
        print(f"🔍 Question: {request.question}")
        sql = generate_sql_with_gemini(request.question, optimizer)
        print(f"📝 Generated SQL: {sql}")
        
        # 2. Validate SQL
        is_valid, error_msg = validate_sql(sql)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid SQL: {error_msg}")
        
        # 3. Execute SQL
        try:
            result_df = con.execute(sql).fetchdf()
            data = result_df.to_dict('records')
            print(f"✅ Query executed: {len(data)} rows")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SQL execution error: {str(e)}")
        
        # 4. Format response with structured output
        formatted_answer, structured_data = format_response_with_gemini(request.question, data, sql)
        
        # 5. Prepare response
        response = QueryResponse(
            question=request.question,
            answer=formatted_answer,
            sql=sql if request.include_sql else None,
            data=data if request.include_data else None,
            metadata={
                "row_count": len(data),
                "columns": list(result_df.columns) if len(data) > 0 else []
            },
            structured=structured_data  # Add structured JSON
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/schema")
async def get_schema():
    """Get database schema"""
    return {
        "tables": optimizer['schema']['tables'],
        "business_glossary": optimizer['business_glossary']
    }

@app.get("/examples")
async def get_examples():
    """Get example questions"""
    examples_by_category = {}
    for example in optimizer['examples']:
        category = example['category']
        if category not in examples_by_category:
            examples_by_category[category] = []
        examples_by_category[category].append({
            'id': example['id'],
            'question': example['question'],
            'category': example['category']
        })
    
    return {
        "total": len(optimizer['examples']),
        "categories": examples_by_category
    }

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    stats = {}
    
    # Get row counts
    for table in ['dim_date', 'dim_branch', 'dim_product', 
                  'fact_registration', 'fact_contract', 'fact_inventory']:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = {"row_count": count}
    
    # Date range
    min_date = con.execute("SELECT MIN(date_key) FROM dim_date").fetchone()[0]
    max_date = con.execute("SELECT MAX(date_key) FROM dim_date").fetchone()[0]
    
    stats['date_range'] = {
        'min': min_date,
        'max': max_date
    }
    
    # Business metrics
    total_registrations = con.execute("SELECT SUM(reg_count) FROM fact_registration").fetchone()[0]
    total_contracts = con.execute("SELECT SUM(contract_count) FROM fact_contract").fetchone()[0]
    total_revenue = con.execute("SELECT SUM(contract_value) FROM fact_contract").fetchone()[0]
    
    stats['business_metrics'] = {
        'total_registrations': float(total_registrations) if total_registrations else 0,
        'total_contracts': float(total_contracts) if total_contracts else 0,
        'total_revenue': float(total_revenue) if total_revenue else 0,
        'conversion_rate': round((total_contracts / total_registrations * 100), 2) if total_registrations else 0
    }
    
    return stats

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
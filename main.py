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
# Business Thresholds (ค่ามาตรฐานทางธุรกิจ)
# ==========================================

BUSINESS_THRESHOLDS = {
    'conversion_rate': {
        'excellent': 90,
        'good': 85,
        'acceptable': 80,
        'poor': 80
    },
    'stock_coverage': {
        'overstock': 1.5,
        'optimal_max': 1.2,
        'optimal_min': 0.8,
        'shortage': 0.5
    },
    'shortage': {
        'critical': 5000,
        'high': 1000,
        'medium': 500,
        'low': 100
    },
    'avg_price': 40000
}

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
    con = duckdb.connect(':memory:')
    
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
    sql_text = re.sub(r'```sql\n?', '', sql_text)
    sql_text = re.sub(r'```\n?', '', sql_text)
    sql_text = re.sub(r'^(SQL:|Query:)\s*', '', sql_text, flags=re.IGNORECASE)
    sql_text = sql_text.strip()
    return sql_text

def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL query"""
    sql_upper = sql.upper()
    
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Dangerous operation detected: {keyword}"
    
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    valid_tables = ['dim_date', 'dim_branch', 'dim_product', 
                   'fact_registration', 'fact_contract', 'fact_inventory']
    
    has_valid_table = any(table in sql for table in valid_tables)
    if not has_valid_table:
        return False, "No valid table found in query"
    
    return True, ""

def find_relevant_examples(question: str, optimizer: dict, max_examples: int = 15) -> list:
    """Find most relevant SQL examples based on the question"""
    question_lower = question.lower()
    all_examples = optimizer['examples']
    
    keywords_map = {
        'demand_high_stock_low': ['demand', 'stock', 'สูง', 'ต่ำ', 'gap', 'ช่องว่าง', 'ไม่พอ'],
        'branch_analysis': ['สาขา', 'branch', 'ร้าน', 'shop'],
        'model_analysis': ['รุ่น', 'model', 'iphone'],
        'registration': ['ลงทะเบียน', 'registration', 'รอ', 'waiting'],
        'stock_supply': ['สต็อค', 'stock', 'supply', 'inventory', 'คงเหลือ'],
        'contract': ['สัญญา', 'contract', 'ทำสัญญา'],
        'ratio': ['ratio', 'อัตราส่วน', 'เปรียบเทียบ'],
        'conversion': ['conversion', 'แปลง', '→'],
        'lost_sales': ['lost', 'สูญเสีย', 'หาย', 'shortage'],
        'efficiency': ['ประสิทธิภาพ', 'efficiency', 'matching', 'จับคู่'],
        'top_performer': ['มากที่สุด', 'สูงสุด', 'most', 'highest', 'top'],
        'shortage': ['ขาด', 'shortage', 'ไม่พอ', 'not enough'],
    }
    
    scored_examples = []
    
    for ex in all_examples:
        score = 0
        ex_question = ex['question'].lower()
        ex_category = ex.get('category', '')
        ex_patterns = ' '.join(ex.get('key_patterns', [])).lower()
        
        if question_lower in ex_question or ex_question in question_lower:
            score += 100
        
        question_words = set(question_lower.split())
        ex_words = set(ex_question.split())
        common_words = question_words & ex_words
        score += len(common_words) * 10
        
        for category, keywords in keywords_map.items():
            question_has_keyword = any(kw in question_lower for kw in keywords)
            example_has_keyword = any(kw in ex_question or kw in ex_patterns for kw in keywords)
            
            if question_has_keyword and example_has_keyword:
                score += 30
        
        if 'demand_supply_analysis' in ex_category:
            if any(kw in question_lower for kw in ['demand', 'stock', 'supply', 'สต็อค', 'ดีมานด์']):
                score += 20
        
        if 'gap' in question_lower or 'ช่องว่าง' in question_lower or 'ไม่พอ' in question_lower:
            if 'gap analysis' in ex_patterns or 'shortage' in ex_patterns:
                score += 50
        
        if 'พร้อมจำนวน' in question_lower or 'with value' in ex_patterns:
            if 'TOP 1 with value' in ex_patterns or 'include COUNT in SELECT' in ex_patterns:
                score += 40
        
        if ex['id'] >= 31 and score > 0:
            score += 15
        
        scored_examples.append((score, ex))
    
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    relevant_examples = [ex for score, ex in scored_examples if score > 0][:max_examples]
    
    if len(relevant_examples) < 5:
        for ex in all_examples:
            if ex['id'] >= 31 and ex not in relevant_examples:
                relevant_examples.append(ex)
                if len(relevant_examples) >= 10:
                    break
    
    if len(relevant_examples) < 5:
        for ex in all_examples[:10]:
            if ex not in relevant_examples:
                relevant_examples.append(ex)
                if len(relevant_examples) >= 10:
                    break
    
    return relevant_examples[:max_examples]

def generate_sql_with_gemini(question: str, optimizer: dict) -> str:
    """Generate SQL using Gemini with relevant examples"""
    current_date = con.execute("SELECT MAX(date_key) FROM fact_inventory").fetchone()[0]
    relevant_examples = find_relevant_examples(question, optimizer, max_examples=15)
    
    example_ids = [ex['id'] for ex in relevant_examples]
    print(f"🔍 Selected examples for '{question}': {example_ids}")
    
    prompt = f"""You are an expert SQL query generator for a DuckDB database.

DATABASE SCHEMA:
{json.dumps(optimizer['schema']['tables'], indent=2, ensure_ascii=False)}

BUSINESS RULES:
{json.dumps(optimizer['business_glossary'], indent=2, ensure_ascii=False)}

DEFAULT ASSUMPTIONS:
{json.dumps(optimizer['default_assumptions'], indent=2, ensure_ascii=False)}

RELEVANT EXAMPLE QUESTIONS AND SQL (FOLLOW THESE PATTERNS EXACTLY!):
{json.dumps(relevant_examples, indent=2, ensure_ascii=False)}

IMPORTANT NOTES:
- Current latest date in database: {current_date}
- Use date_key format: YYYYMMDD (e.g., 20251227)
- For "today" or "วันนี้", use: WHERE date_key = {current_date}
- For "this month" or "เดือนนี้", use: WHERE date_key BETWEEN {str(current_date)[:6]}01 AND {str(current_date)[:6]}31
- For inventory queries, use latest date: WHERE date_key = (SELECT MAX(date_key) FROM fact_inventory)
- Only shops (BR001-BR005) have registrations and contracts
- Warehouse (BR000) only appears in inventory

CRITICAL INSTRUCTIONS:
1. Find the MOST SIMILAR example above
2. Use the EXACT same SQL pattern as that example
3. Only change the specific filters/values needed for this question
4. Do NOT invent new SQL patterns or structures
5. If asking for "มากที่สุด" (most/highest), make sure to SELECT both the name AND the count/value

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

def analyze_business_context(question: str, data: List[Dict], sql: str) -> Dict:
    """
    Pre-analyze data to provide business context and actionable recommendations
    """
    if not data:
        return {
            "query_type": "NO_DATA",
            "context": "ไม่พบข้อมูลที่ตรงกับเงื่อนไข",
            "recommendations": [],
            "priority": "ต่ำ",
            "timeline": "ไม่ต้องดำเนินการ"
        }
    
    context = {
        "query_type": "GENERAL",
        "key_finding": None,
        "recommendations": [],
        "priority": "ปานกลาง",
        "timeline": "1 สัปดาห์",
        "financial_impact": None,
        "next_steps": []
    }
    
    first_row = data[0]
    q_lower = question.lower()
    
    # ORDER RECOMMENDATION
    if any(kw in q_lower for kw in ['order', 'สั่ง', 'เพิ่ม', 'ควร']):
        context["query_type"] = "ORDER_RECOMMENDATION"
        
        if 'stock_shortage' in first_row and 'conversion_rate' in first_row:
            shortage = float(first_row.get('stock_shortage', 0))
            conversion = float(first_row.get('conversion_rate', 0))
            model = first_row.get('model_name', 'รุ่นนี้')
            demand = float(first_row.get('demand', 0))
            
            if shortage > 0 and conversion >= BUSINESS_THRESHOLDS['conversion_rate']['acceptable']:
                if shortage >= BUSINESS_THRESHOLDS['shortage']['critical']:
                    context["priority"] = "🔴 สูงมาก - URGENT"
                    context["timeline"] = "24 ชั่วโมง"
                elif shortage >= BUSINESS_THRESHOLDS['shortage']['high']:
                    context["priority"] = "🟠 สูง"
                    context["timeline"] = "2-3 วัน"
                else:
                    context["priority"] = "🟡 ปานกลาง"
                    context["timeline"] = "1 สัปดาห์"
                
                shortage_percent = (shortage / demand * 100) if demand > 0 else 0
                lost_sales_value = shortage * BUSINESS_THRESHOLDS['avg_price'] * (conversion / 100)
                
                context["key_finding"] = f"{model} มี shortage {shortage:,.0f} เครื่อง ({shortage_percent:.0f}% ของ demand)"
                context["financial_impact"] = {
                    "shortage_units": f"{shortage:,.0f} เครื่อง",
                    "lost_sales_risk": f"฿{lost_sales_value:,.0f}",
                    "note": f"มูลค่าที่อาจเสียหากไม่ดำเนินการภายใน {context['timeline']}"
                }
                
                context["recommendations"] = [
                    f"✅ ต้อง order {model} เพิ่ม {shortage:,.0f} เครื่อง",
                    f"Priority: {context['priority']}",
                    f"Timeline: {context['timeline']}",
                    f"เหตุผล: Conversion {conversion:.1f}% ดี แต่ stock ขาด {shortage_percent:.0f}%"
                ]
                
                context["next_steps"] = [
                    "1. ตรวจสอบ supplier lead time",
                    "2. Review budget approval",
                    f"3. Confirm order: {shortage:,.0f} เครื่อง",
                    "4. กำหนด delivery date",
                    "5. แจ้ง Sales team"
                ]
    
    # CONVERSION ANALYSIS
    elif 'conversion' in q_lower:
        context["query_type"] = "CONVERSION_ANALYSIS"
        
        if 'conversion_rate' in first_row:
            conversion = float(first_row.get('conversion_rate', 0))
            entity = first_row.get('model_name', first_row.get('branch_name', 'รายการนี้'))
            
            if conversion < BUSINESS_THRESHOLDS['conversion_rate']['acceptable']:
                context["priority"] = "🔴 สูง"
                context["timeline"] = "3-5 วัน"
                gap = BUSINESS_THRESHOLDS['conversion_rate']['acceptable'] - conversion
                
                context["key_finding"] = f"{entity} conversion {conversion:.1f}% ต่ำกว่าเป้า"
                context["recommendations"] = [
                    f"⚠️ Conversion {conversion:.1f}% ต่ำกว่า 80% ({gap:.1f}% gap)",
                    "ตรวจสอบ: ราคา/เงื่อนไข/Sales skill",
                    "Target: ยกระดับให้ถึง 80%",
                    f"Timeline: {context['timeline']}"
                ]
                
                context["next_steps"] = [
                    "1. วิเคราะห์สาเหตุ conversion ต่ำ",
                    "2. Survey ลูกค้าที่ไม่ทำสัญญา",
                    "3. Training Sales team",
                    "4. ติดตามผลทุก 3 วัน"
                ]
    
    # STOCK COVERAGE
    elif 'coverage' in sql.lower() or ('stock' in q_lower and 'demand' in q_lower):
        context["query_type"] = "STOCK_COVERAGE"
        
        if 'stock_coverage' in first_row:
            coverage = float(first_row.get('stock_coverage', 0))
            entity = first_row.get('model_name', 'รายการนี้')
            
            if coverage >= BUSINESS_THRESHOLDS['stock_coverage']['overstock']:
                context["priority"] = "🟡 ปานกลาง"
                context["recommendations"] = [
                    f"⚠️ Stock Coverage {coverage:.1f}x (มากเกิน)",
                    "พิจารณาทำ Promotion",
                    "ระวัง Obsolescence risk"
                ]
            elif coverage <= BUSINESS_THRESHOLDS['stock_coverage']['shortage']:
                context["priority"] = "🔴 สูงมาก"
                context["recommendations"] = [
                    f"🔴 Stock Coverage {coverage:.1f}x (ขาดมาก)",
                    "Order เพิ่มทันที",
                    "ติดต่อ supplier เร่งส่ง"
                ]
    
    return context

def format_response_with_gemini(question: str, data: List[Dict], sql: str) -> tuple[str, Optional[Dict]]:
    """Format response with business context"""
    
    business_context = analyze_business_context(question, data, sql)
    
    prompt = f"""Analyze this data with business context and create actionable insights.

USER QUESTION: {question}
QUERY RESULTS: {json.dumps(data[:5], indent=2, ensure_ascii=False)}

BUSINESS CONTEXT:
{json.dumps(business_context, indent=2, ensure_ascii=False)}

Return JSON:
{{
    "summary": "คำตอบสั้นๆ พร้อมตัวเลข",
    "key_metrics": [{{"label": "...", "value": "...", "unit": "..."}}],
    "actionable_recommendation": {{
        "action": "{business_context.get('recommendations', [''])[0] if business_context.get('recommendations') else 'ดูรายละเอียด'}",
        "priority": "{business_context.get('priority', 'ปานกลาง')}",
        "timeline": "{business_context.get('timeline', '1 สัปดาห์')}",
        "reason": "เหตุผลสั้นๆ"
    }},
    "details": ["รายละเอียด..."],
    "next_steps": {json.dumps(business_context.get('next_steps', ['ติดตาม']))}
}}

Return ONLY valid JSON, no markdown:"""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        structured_data = json.loads(response_text)
        
        if not all(key in structured_data for key in ['summary', 'key_metrics', 'actionable_recommendation']):
            raise ValueError("Missing keys")
        
        markdown_parts = []
        markdown_parts.append(f"**📊 {structured_data['summary']}**\n")
        
        if structured_data.get('key_metrics'):
            markdown_parts.append("**📈 ตัวเลขสำคัญ:**")
            for metric in structured_data['key_metrics']:
                markdown_parts.append(f"• **{metric['label']}:** {metric['value']} {metric['unit']}")
            markdown_parts.append("")
        
        if structured_data.get('actionable_recommendation'):
            rec = structured_data['actionable_recommendation']
            markdown_parts.append("**💡 คำแนะนำที่ใช้ได้จริง:**")
            markdown_parts.append(f"• **Action:** {rec.get('action', 'ดูรายละเอียด')}")
            markdown_parts.append(f"• **Priority:** {rec.get('priority', 'ปานกลาง')}")
            markdown_parts.append(f"• **Timeline:** {rec.get('timeline', '1 สัปดาห์')}")
            markdown_parts.append(f"• **เหตุผล:** {rec.get('reason', 'ดูข้อมูล')}")
            markdown_parts.append("")
        
        if structured_data.get('details'):
            markdown_parts.append("**📋 รายละเอียด:**")
            for detail in structured_data['details']:
                markdown_parts.append(f"• {detail}")
            markdown_parts.append("")
        
        if structured_data.get('next_steps'):
            markdown_parts.append("**🎯 ขั้นตอนถัดไป:**")
            for step in structured_data['next_steps']:
                markdown_parts.append(step)
            markdown_parts.append("")
        
        if business_context.get('financial_impact'):
            fi = business_context['financial_impact']
            markdown_parts.append("**💰 ผลกระทบทางการเงิน:**")
            if 'shortage_units' in fi:
                markdown_parts.append(f"• Shortage: {fi['shortage_units']}")
            if 'lost_sales_risk' in fi:
                markdown_parts.append(f"• Lost Sales Risk: {fi['lost_sales_risk']}")
            if 'note' in fi:
                markdown_parts.append(f"• {fi['note']}")
        
        formatted_markdown = '\n'.join(markdown_parts)
        return formatted_markdown, structured_data
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        fallback = f"**📊 สรุป:** {business_context.get('key_finding', f'พบข้อมูล {len(data)} รายการ')}"
        if business_context.get('recommendations'):
            fallback += "\n\n**💡 คำแนะนำ:**"
            for rec in business_context['recommendations'][:3]:
                fallback += f"\n• {rec}"
        return fallback, None

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "iPhone Demand-Supply Chatbot API",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini not configured")
    
    try:
        print(f"🔍 Question: {request.question}")
        sql = generate_sql_with_gemini(request.question, optimizer)
        print(f"📝 SQL: {sql}")
        
        is_valid, error_msg = validate_sql(sql)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid SQL: {error_msg}")
        
        result_df = con.execute(sql).fetchdf()
        data = result_df.to_dict('records')
        print(f"✅ {len(data)} rows")
        
        formatted_answer, structured_data = format_response_with_gemini(request.question, data, sql)
        
        return QueryResponse(
            question=request.question,
            answer=formatted_answer,
            sql=sql if request.include_sql else None,
            data=data if request.include_data else None,
            metadata={
                "row_count": len(data),
                "columns": list(result_df.columns) if len(data) > 0 else []
            },
            structured=structured_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/schema")
async def get_schema():
    return {
        "tables": optimizer['schema']['tables'],
        "business_glossary": optimizer['business_glossary']
    }

@app.get("/examples")
async def get_examples():
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
    stats = {}
    
    for table in ['dim_date', 'dim_branch', 'dim_product', 
                  'fact_registration', 'fact_contract', 'fact_inventory']:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = {"row_count": count}
    
    min_date = con.execute("SELECT MIN(date_key) FROM dim_date").fetchone()[0]
    max_date = con.execute("SELECT MAX(date_key) FROM dim_date").fetchone()[0]
    
    stats['date_range'] = {'min': min_date, 'max': max_date}
    
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")  

class SQLResponse(BaseModel):
    sql: str = Field(description="SQL 쿼리")

class SQLState(TypedDict, total=False):
    question: str
    schema_context: str
    sql: str
    results: List[str]
    source: str #normal / fallback
class EcommerceTextToSQLAgent:
    def __init__(self):
        print("RAG 초기화...")
        self.db = SQLDatabase.from_uri(DATABASE_URL)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory="./chroma_vectors", embedding_function=self.embeddings)
        
        self._init_schema_vectors()

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            self.sql_chain = self._create_sql_chain()

        # LangGraph 워크플로우 초기화
        self.graph = self._create_graph()

        print("RAG 준비 완료 !!!!")
       
    # Vector store에 DB 스키마 저장
    # def _init_schema_vectors(self):
    #     schema_text = self._get_schema()
    #     tables = self.db.get_usable_table_names()
        
    #     schema_texts = [
    #         schema_text,
    #         "Ecommerce DB: 상품(item), 옵션(item_option), 쿠폰(coupon), 구매(purchase_detail), 구매상품조인(purchase_item), 리뷰(review), 유저(user)",
    #         "INSERT 예시: INSERT INTO item (item_name, item_price) VALUES ('상품명', 10000)",
    #         "Join 예시: SELECT i.item_name, SUM(pd.quantity) FROM item i JOIN purchase_detail pd"
    #     ]
    #     self.vector_store.add_texts(schema_texts) #TODO: embeding시 테이블 메타 정보 매핑
    #     print(f"DB 스키마 벡터화: {len(tables)} 테이블")

    # Vector store에 DB 스키마 저장
    def _init_schema_vectors(self):
        schema_text = self._get_schema()
        tables = self.db.get_usable_table_names()
        
        table_meta_mapping = {
            "coupon": {
                "desc": "쿠폰 관리 테이블",
                "columns": "coupon_name(쿠폰명), discount_rate(할인율%), started_date(시작일!), ended_date(종료일!), user_id(유저ID), created_date(생성일), used_date(사용일)",
                "keywords": ["쿠폰", "할인", "시작일", "종료일", "프로모션", "쿠폰명", "할인율"],
                "example": "INSERT INTO coupon (coupon_name, discount_rate, started_date, ended_date, user_id) VALUES ('크리스마스', 20, '2025-12-25 00:00:00', '2025-12-31 23:59:59', 1);"
            },
            "item": {
                "desc": "상품 기본 정보 테이블",
                "columns": "item_id(PK), item_name(상품명!), item_price(가격), item_image_url(이미지URL), description(설명), deleted_status(삭제여부)",
                "keywords": ["상품", "제품", "아이템", "가격", "이미지", "설명"],
                "example": "INSERT INTO item (item_name, item_price) VALUES ('티셔츠', 15000);"
            },
            "item_option": {
                "desc": "상품 옵션 테이블 (색상, 사이즈, 재고)",
                "columns": "i_option_id(PK), i_option_name(옵션명!), i_option_quantity(재고량), item_id(상품ID), deleted_status(삭제여부)",
                "keywords": ["옵션", "색상", "사이즈", "재고", "옵션명", "수량"],
                "example": "INSERT INTO item_option (i_option_name, i_option_quantity, item_id) VALUES ('빨강-M', 100, 1);"
            },
            "purchase_detail": {
                "desc": "구매 주문 내역 테이블 (배송상태 포함)",
                "columns": "purchase_id(PK), user_id(구매자ID), quantity(구매수량), purchase_date(구매일), delivery_status(배송상태: BEFORE_DELIVERY|COMPLETED|...)",
                "keywords": ["구매", "주문", "배송", "수량", "배송상태"],
                "example": "SELECT * FROM purchase_detail WHERE delivery_status='COMPLETED';"
            },
            "purchase_item": {
                "desc": "구매-옵션 다대다 조인 테이블",
                "columns": "purchase_id(FK), option_id(FK) (복합PK)",
                "keywords": ["구매상품", "주문옵션", "구매내역"],
                "example": "SELECT pi.*, io.i_option_name FROM purchase_item pi JOIN item_option io ON pi.option_id=io.i_option_id;"
            },
            "review": {
                "desc": "상품 리뷰 테이블",
                "columns": "review_id(PK), review_score(별점 1-5), comment(리뷰내용), item_id(상품ID), purchase_id(구매ID), deleted_status(삭제여부)",
                "keywords": ["리뷰", "후기", "별점", "평점", "댓글"],
                "example": "INSERT INTO review (review_score, comment, item_id, purchase_id) VALUES (5, '좋아요', 1, 1);"
            },
            "user": {
                "desc": "회원 테이블",
                "columns": "user_id(PK), email(이메일), password(비밀번호), user_role(USER|ADMIN), address_road(도로명주소), address_detail(상세주소), deleted_status(삭제여부)",
                "keywords": ["유저", "회원", "사용자", "email", "이메일", "주소"],
                "example": "SELECT user_id FROM user WHERE email LIKE '%user1%';"
            }
        }
        
        schema_texts = [schema_text]
        
        for table_name, meta in table_meta_mapping.items():
            if table_name in tables:
                meta_text = f"""📋 {table_name} 테이블 ({meta['desc']})

                [주요 컬럼]:
                {meta['columns']}

                [검색 키워드]: {', '.join(meta['keywords'])}

                [사용 예시]:
                {meta['example']}

                [주의사항]:
                - AUTO_INCREMENT 컬럼(coupon_id, item_id 등)은 생략
                - created_date, modified_date는 자동 생성
                - user는 email로 검색 (user_name 컬럼 없음)"""

        schema_texts.append(meta_text)
        
        common_patterns = [
            "조인 예시: SELECT i.item_name, pd.quantity FROM item i JOIN purchase_detail pd ON i.item_id=pd.user_id;",
            "통계 예시: SELECT AVG(review_score), COUNT(*) FROM review GROUP BY item_id;",
            "날짜 예시: '2025-12-25 00:00:00' 또는 NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY)",
            "Enum 예시: delivery_status IN ('BEFORE_DELIVERY', 'COMPLETED', 'DELIVERY_IN_PROGRESS')"
        ]
        schema_texts.extend(common_patterns)
        
        self.vector_store.add_texts(schema_texts)
        print(f"DB 스키마 벡터화 완료!")
        print(f"테이블: {len(tables)}개")
        print(f"메타 문서: {len(schema_texts)-1}개")


    # DB 스키마 가져오기
    def _get_schema(self):
        tables = self.db.get_usable_table_names()
        schema_info = []
        
        for table in tables:
            columns = self.db.get_table_info([table])
            schema_info.append(f"{table}: {columns}") #TODO: 프롬프트로도 많이 함.
        
        return "\n".join(schema_info)
    
    # Few shot
    def _create_sql_chain(self):

        base_parser = PydanticOutputParser(pydantic_object=SQLResponse)

        parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=self.llm
        )

        few_shot = """
        예시 1: "티셔츠 10000원 추가" → INSERT INTO item (item_name, item_price) VALUES ('티셔츠', 10000);
        예시 2: "빨간색 옵션 100개" → INSERT INTO item_option (i_option_name, i_option_quantity) VALUES ('빨간색', 100);
        예시 3: "상품 목록" → SELECT * FROM item LIMIT 10;
        예시 4: "총 매출" → SELECT SUM(item_price * quantity) FROM item i JOIN purchase_detail pd ON i.item_id = pd.item_id;
        예시 5: "블랙프라이데이 쿠폰" → INSERT INTO coupon (coupon_name, discount_rate, started_date, ended_date, user_id) VALUES ('크리스마스', 20, '2025-12-25 00:00:00', '2025-12-31 23:59:59', 1);
        예시 6: "user1 쿠폰" → INSERT INTO coupon (coupon_name, discount_rate, started_date, ended_date, user_id) VALUES ('테스트', 10, NOW(), DATE_ADD(NOW(), INTERVAL 7 DAY), (SELECT user_id FROM user WHERE email='user1@example.com'));
        """ 
                
        template = """Ecommerce MySQL 데이터베이스입니다.

        [실시간 DB 스키마]:
        {schema_context}

        [사용 예시]:
        """ + few_shot + """

        [사용자 질문]:
        {question}

        [출력 형식]:
        출력은 반드시 JSON 형식
        형식:
        {{
        "sql": "SQL문;"
        }}

        [출력 형식 엄수]:
        1. 한 줄 SQL만 출력 (줄바꿈 X)
        2. 맨 끝에 세미콜론(;) 하나만 (세미콜론 여러개 X)
        3. 백틱(`) 절대 사용 금지
        4. 마크다운(```) 절대 사용 금지
        5. 주석(--) 절대 사용 금지 
        6. 따옴표 등 금지
        7. 완전한 SQL만 한 줄 (끊기면 안됨)
        8.출력은 반드시 JSON 형식

        [지시사항]:
        1. 실제 스키마만 사용 (item, item_option, coupon, purchase, review)
        2. 백틱(`)과 마크다운(```) 절대 사용 금지
        3. AUTO_INCREMENT 생략
        4. datetime 형식: 'YYYY-MM-DD HH:MM:SS'
        5. 복잡 쿼리는 조인/서브쿼리 사용 OK
        6. 모든 테이블의 created_date는 현재 시간으로 사용

        SQL:""" #TODO: 개행이나 텍스트 지시사항 추가
        
        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | parser
        #LCEL 체인 : 프롬프트와 LLM을 | 연산자로 연결하여 작성하여 체인을 구현
        #prompt 한번 더 format할 필요 x
        #TODO: pydeantic parser로 (범용성이 크다)
    
    
    def _safe_execute(self, clean_sql: str):
        results = []
        for sql_query in clean_sql.split(';'):
            sql_query = sql_query.strip()
            if sql_query:
                try:
                    result = self.db.run(sql_query)
                    results.append(f"{result}")
                    print(f"SQL Query 실행: {sql_query[:50]}...")
                except Exception as e:
                    results.append(f"{str(e)[:50]}...")
                    print(f"SQL Query 실행 실패: {sql_query[:50]}...")
        return results
    
    def execute_query(self, natural_query: str): 
        try:
            # LangGraph 초기 상태
            init_state: SQLState = {
                "question": natural_query
            }

            # graph.invoke 사용 (단일 실행)
            final_state: SQLState = self.graph.invoke(init_state)

            rag_context = final_state.get("schema_context", "")
            sql = final_state.get("sql", "")
            results = final_state.get("results", [])
            source = final_state.get("source", "normal")

            return {
                "status": "success",
                "result": {
                    "query": natural_query,
                    "rag_context": rag_context[:200],
                    "sql": sql,
                    "results": results,
                    "source": source,
                },
            }

        except Exception as e:
            print(f"error message: {str(e)}")
            return {"status": "error", "error": str(e)}

    # LangGraph node: 스키마 검색
    def _node_retrieve_schema(self, state: SQLState) -> SQLState:
        question = state["question"]
        relevant_docs = self.vector_store.similarity_search(question, k=2)
        rag_context = "\n".join([doc.page_content for doc in relevant_docs])
        print(f"RAG 스키마: {rag_context[:100]}...")
        return {
            **state,
            "schema_context": rag_context,
        }
    def _node_generate_sql(self, state: SQLState) -> SQLState:
        if not (self.llm and self.sql_chain):
            return {**state, "source": "fallback"}
        
        try:
            invoke_result = self.sql_chain.invoke({
                "schema_context": state.get("schema_context", ""),
                "question": state["question"],
            })
            sql = str(invoke_result.sql).strip()
            print(f"SQL: {repr(sql)}...")
            return {
                **state,
                "sql": sql,
                "source": "normal",
            }
        except Exception as e:
            print(f"SQL 생성 실패: {str(e)}")
            return {
                **state,
                "source": "fallback",
            }
        
    # LangGraph node: SQL 실행 및 fallback 처리
    def _node_run_sql(self, state: SQLState) -> SQLState:
        source = state.get("source", "normal")
        sql = state.get("sql", "").strip()

        # LLM 정상 SQL 없는 경우 → fallback SQL 생성
        if source == "fallback" or not sql:
            print("Fallback 모드 실행...")
            fallback_sql = self._generate_fallback_sql(state["question"])
            results = self._safe_execute(fallback_sql)
            return {
                **state,
                "sql": fallback_sql,
                "results": results,
                "source": "fallback",
            }

        # 정상 SQL 실행
        results = self._safe_execute(sql)
        return {
            **state,
            "results": results,
        }
    
    def _create_graph(self):
        builder = StateGraph(SQLState)

        # 노드 등록
        builder.add_node("retrieve_schema", self._node_retrieve_schema)
        builder.add_node("generate_sql", self._node_generate_sql)
        builder.add_node("run_sql", self._node_run_sql)

        # 엣지 연결
        builder.add_edge(START, "retrieve_schema")
        builder.add_edge("retrieve_schema", "generate_sql")
        builder.add_edge("generate_sql", "run_sql")
        builder.add_edge("run_sql", END)

        # 그래프 컴파일
        app = builder.compile()
        return app

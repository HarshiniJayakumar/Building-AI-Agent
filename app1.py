import json
import os
import re
import string
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="AI Agent - Milestone 2 (Groq)",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Customer Support AI Agent")
st.caption(
    "Groq-powered agent with Python tool calling, missing-data safeguards, "
    "and a local fallback."
)
st.write("Functions: Check order status, search policies, and calculate shipping fees.")

# ================= API CONFIGURATION =================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Use one supported model consistently for both tool selection and final response.
# Confirm current availability in your Groq Console if the provider changes model IDs.
MODEL_NAME = "openai/gpt-oss-20b"

client = Groq(api_key=API_KEY) if API_KEY else None


# ================= 1. PYTHON TOOLS =================
def check_order_status(order_id: str) -> str:
    """Look up a sample order by its ID."""
    database = {
        "ORD1001": "Status: Shipped | Expected Delivery: 28 Oct 2026",
        "ORD1002": "Status: Processing in warehouse | Expected Dispatch: 26 Oct 2026",
        "ORD1003": "Status: Out for Delivery | Expected Delivery: Today",
        "ORD1004": "Status: Delivered | Delivered on: 20 Oct 2026",
        "ORD1005": "Status: Cancelled | Refund initiated to original payment method",
        "ORD1006": "Status: Pending Payment | Awaiting bank confirmation",
        "ORD1007": "Status: Packed | Ready for courier pickup",
        "ORD1008": "Status: Delayed due to weather conditions | Expected Delivery: 30 Oct 2026",
    }

    cleaned_id = re.sub(r"[^A-Za-z0-9]", "", str(order_id)).upper()
    return database.get(
        cleaned_id,
        "NOT_FOUND: Order ID does not exist in our database.",
    )


def search_policy(topic: str) -> str:
    """Retrieve a policy from the local sample policy database."""
    policies = {
        "return": "Unused items in original packaging can be returned within 7 days of delivery.",
        "refund": "Refunds are processed to the original payment method within 3-5 business days after inspection.",
        "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
        "warranty": "All electronic items come with a 1-year manufacturer warranty covering internal defects.",
        "cancellation": "Orders can be cancelled free of charge within 2 hours of placement.",
        "exchange": "Product exchanges are permitted for different sizes or colors within 7 days of delivery.",
        "support": "Customer support is available via live chat 24/7 and via phone Mon-Fri 9 AM - 6 PM.",
        "discount": "First-time buyers can use code WELCOME10 for a 10% discount on their order.",
    }

    topic_lower = str(topic).lower().strip()
    for key, value in policies.items():
        if key in topic_lower:
            return f"Policy [{key.upper()}]: {value}"

    return "NO_MATCH: Policy not found for this topic."


def calculate_shipping_fee(weight_kg: float) -> str:
    """Calculate shipping charges: Rs. 50 base fee + Rs. 20 per kg."""
    try:
        weight = float(weight_kg)
    except (TypeError, ValueError):
        return "INVALID_INPUT: Please provide the package weight in kilograms."

    if weight <= 0:
        return "INVALID_INPUT: Package weight must be greater than zero."

    fee = 50 + (weight * 20)
    return f"The shipping fee for {weight:g} kg is Rs. {fee:.2f}."


AVAILABLE_TOOLS = {
    "check_order_status": check_order_status,
    "search_policy": search_policy,
    "calculate_shipping_fee": calculate_shipping_fee,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Look up the status of a specific order in the sample order database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID such as ORD1001",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve a supported return, refund, shipping, warranty, cancellation, exchange, support, or discount policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The policy topic, such as return or warranty",
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping_fee",
            "description": "Calculate shipping fee using Rs. 50 + Rs. 20 per kilogram.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "Package weight in kilograms; must be greater than zero",
                    }
                },
                "required": ["weight_kg"],
                "additionalProperties": False,
            },
        },
    },
]


# ================= 2. LOCAL FALLBACK =================
def local_fallback_router(question: str) -> str:
    """Answer only questions that can be safely resolved from local sample data."""
    question_text = question.strip()
    ctx = question_text.lower()

    # Handle explicit weight-based calculations first.
    shipping_calculation_intent = (
        ("shipping" in ctx or "delivery fee" in ctx or "postage" in ctx)
        and any(word in ctx for word in ("fee", "cost", "charge", "calculate", "price", "kg", "kilogram"))
    )

    if shipping_calculation_intent:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)\b", ctx)
        if not match:
            return (
                "[LOCAL FALLBACK] Please specify the package weight in kilograms "
                "(for example, 'Calculate shipping fee for 3.5 kg')."
            )

        weight = float(match.group(1))
        return f"[LOCAL FALLBACK] {calculate_shipping_fee(weight)}"

    # Match order IDs without trailing punctuation.
    if "order" in ctx or re.search(r"\bord\d+\b", ctx):
        order_match = re.search(r"\b(ORD\d+)\b", question_text, re.IGNORECASE)
        if order_match:
            return f"[LOCAL FALLBACK] {check_order_status(order_match.group(1))}"
        return "[LOCAL FALLBACK] Please provide an order ID, such as ORD1001."

    # Use known policy entries only.
    policy_keywords = [
        "cancellation",
        "cancel",
        "exchange",
        "warranty",
        "return",
        "refund",
        "shipping",
        "support",
        "discount",
    ]
    for keyword in policy_keywords:
        if keyword in ctx:
            topic = "cancellation" if keyword == "cancel" else keyword
            return f"[LOCAL FALLBACK] {search_policy(topic)}"

    return (
        "[LOCAL FALLBACK] I cannot answer this from the available local order "
        "and policy data. Please try again when the Groq API is available."
    )


# ================= 3. GROQ AGENT / TOOL-CALL LOOP =================
SYSTEM_PROMPT = """
You are a customer-support assistant for a small demonstration store.
Use the supplied Python tools for questions about orders, policies, and shipping calculations.
Tool outputs and the local sample data are the only source of truth for those topics.
Never invent order statuses, policies, delivery dates, or prices.
If a tool returns NOT_FOUND, NO_MATCH, or INVALID_INPUT, explain that result clearly.
If the question is unrelated to the available tools or data, say that the information is unavailable.
Do not claim that you checked a real external system; these are sample local records.
Keep answers concise and clear.
""".strip()


def ask_agent(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."

    if client is None:
        return (
            "❌ GROQ_API_KEY is missing. Add it to your .env file. "
            "For supported questions, local fallback is available below.\n\n"
            + local_fallback_router(question)
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]

    try:
        # Allow a few rounds in case the model requests more than one tool.
        for _ in range(4):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                temperature=0,
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                answer = assistant_message.content
                if answer and answer.strip():
                    return answer.strip()

                return local_fallback_router(question)

            # Append the assistant tool-call message in the API-compatible format.
            messages.append(assistant_message.model_dump(exclude_none=True))

            for tool_call in tool_calls:
                function_name = tool_call.function.name

                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    function_args = {}

                tool_function = AVAILABLE_TOOLS.get(function_name)
                if tool_function is None:
                    tool_output = "TOOL_ERROR: Requested tool is not available."
                else:
                    try:
                        tool_output = str(tool_function(**function_args))
                    except (TypeError, ValueError) as tool_error:
                        tool_output = (
                            "INVALID_INPUT: The tool arguments were invalid. "
                            f"Details: {tool_error}"
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

        # Do not silently make up an answer if the model keeps requesting tools.
        return (
            "I could not complete the request after several tool calls. "
            "Please rephrase your question and try again."
        )

    except Exception as error:
        # Keep the app usable during API/network errors, but do not fabricate data.
        fallback_answer = local_fallback_router(question)
        return (
            f"⚠️ Groq API request failed ({type(error).__name__}). "
            f"Trying the local fallback:\n\n{fallback_answer}"
        )


# ================= 4. STREAMLIT CHAT UI =================
if "messages" not in st.session_state:
    st.session_state.messages = []


st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input(
    "Ask about orders (ORD1001-ORD1008), policies, or shipping fees..."
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Groq model is processing your question and may call Python tools..."):
        answer = ask_agent(user_query)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("🔁 Consistency Tester")
    st.write("Runs the same question twice and compares the displayed answers.")

    test_question = st.text_input(
        "Test Question:",
        value="What is your warranty policy?",
    )

    if st.button("Run Consistency Test"):
        with st.spinner("Running the same question twice..."):
            result_one = ask_agent(test_question)
            result_two = ask_agent(test_question)

        st.markdown("**Run 1 Output:**")
        st.info(result_one)
        st.markdown("**Run 2 Output:**")
        st.info(result_two)

        if result_one.strip() == result_two.strip():
            st.success("✅ PASS: Both displayed responses are identical.")
        else:
            st.warning(
                "⚠️ The displayed responses differ. This can happen even when "
                "the factual answer is the same."
            )

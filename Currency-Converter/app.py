import streamlit as st
import requests
import os

from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

EXCHANGE_RATE_API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Currency Converter",
    page_icon="💱"
)

st.title("💱 AI Currency Converter")


# ==============================
# MODEL
# ==============================

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    max_new_tokens=1024,
)

model = ChatHuggingFace(llm=llm)


# ==============================
# TOOLS
# ==============================

@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> float:
    """
    Fetch the currency conversion rate.
    """

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    return data["conversion_rate"]


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Convert the given amount using the conversion rate.
    """

    return base_currency_value * conversion_rate


llm_with_tools = model.bind_tools(
    [get_conversion_factor, convert]
)


# ==============================
# UI
# ==============================

amount = st.number_input(
    "Amount",
    min_value=0.01,
    value=100.0
)

col1, col2 = st.columns(2)

with col1:
    from_currency = st.selectbox(
        "From",
        ["INR", "USD", "EUR", "GBP", "JPY"]
    )

with col2:
    to_currency = st.selectbox(
        "To",
        ["USD", "INR", "EUR", "GBP", "JPY"]
    )


# ==============================
# CONVERT BUTTON
# ==============================

if st.button("Convert 💱"):

    if from_currency == to_currency:
        st.warning("Please select different currencies.")

    else:

        messages = [
            HumanMessage(
                content=(
                    f"What is the conversion factor between "
                    f"{from_currency} and {to_currency}, "
                    f"and convert {amount} {from_currency} "
                    f"to {to_currency}."
                )
            )
        ]

        conversion_rate = None

        with st.spinner("Converting..."):

            while True:

                # Ask DeepSeek
                ai_message = llm_with_tools.invoke(messages)

                messages.append(ai_message)

                # Final response
                if not ai_message.tool_calls:
                    break

                # Execute tools
                for tool_call in ai_message.tool_calls:

                    if tool_call["name"] == "get_conversion_factor":

                        result = get_conversion_factor.invoke(
                            tool_call
                        )

                        conversion_rate = float(result.content)

                        messages.append(result)

                    elif tool_call["name"] == "convert":

                        result = convert.invoke({
                            "base_currency_value":
                                tool_call["args"]["base_currency_value"],

                            "conversion_rate":
                                conversion_rate
                        })

                        messages.append(
                            ToolMessage(
                                content=str(result),
                                name="convert",
                                tool_call_id=tool_call["id"]
                            )
                        )


        # ==============================
        # DISPLAY RESULT
        # ==============================

        st.success(
            f"{amount:g} {from_currency} = "
            f"{result} {to_currency}"
        )

        st.info(
            f"Exchange rate: "
            f"1 {from_currency} = "
            f"{conversion_rate} {to_currency}"
        )

        st.write("### 🤖 AI Response")

        st.write(messages[-1].content)
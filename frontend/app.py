import requests
import streamlit as st

st.set_page_config(
    page_title="Name Matching & Recipe Chatbot", page_icon="🤖", layout="wide"
)

API_BASE_URL = "http://localhost:8000"

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .task-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .match-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .recipe-card {
        background-color: #f5f5f5;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 5px solid #2ca02c;
        line-height: 1.6;
        font-size: 16px;
        white-space: pre-wrap;
    }
    .best-match {
        background-color: #ffe4b5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ffa500;
        margin-bottom: 1.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def display_task1():
    st.markdown(
        '<div class="task-header">📝 Task 1: Name Matching System</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Find similar names from our dataset using advanced fuzzy matching and semantic similarity."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        name_input = st.text_input(
            "Enter a name to find matches:",
            placeholder="e.g., Gita, Mohammad, Priya, Kris",
            key="name_input",
        )

    with col2:
        st.write("")
        st.write("")
        search_button = st.button(" Find Similar Names", use_container_width=True)

    # Sample names
    st.info(" **Try these examples:** Gita, Mohammad, Prya, Kris, Sandeep, Lakshmi")

    if search_button and name_input:
        with st.spinner("Searching for similar names..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/api/match-names",
                    json={"name": name_input},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display best match
                    st.markdown("### Best Match")
                    best_match = data["best_match"]
                    st.markdown(
                        f"""
                    <div class="best-match">
                        <h2 style="margin: 0; color: #ff6347;">{best_match["name"]}</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                            Similarity Score: <strong>{best_match["score"]:.1%}</strong>
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Display all matches
                    st.markdown("### All Matches (Ranked)")

                    # Create columns for better layout
                    for i, match in enumerate(data["all_matches"], 1):
                        col1, col2, col3 = st.columns([1, 3, 2])

                        with col1:
                            st.markdown(f"**#{i}**")

                        with col2:
                            st.markdown(f"**{match['name']}**")

                        with col3:
                            # Progress bar for score
                            score_pct = match["score"]
                            st.progress(score_pct)
                            st.caption(f"Score: {score_pct:.1%}")

                        if i < len(data["all_matches"]):
                            st.divider()

                    # Additional info
                    st.success(
                        f"Found {len(data['all_matches'])} matching names for '{name_input}'"
                    )

                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to API. Please ensure the backend server is running."
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif search_button:
        st.warning("Please enter a name to search.")


def display_task2():
    """Display Task 2: Recipe Chatbot"""
    st.markdown(
        '<div class="task-header">🍳 Task 2: Recipe Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.write("Get personalized recipe suggestions based on the ingredients you have!")

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        ingredients_input = st.text_area(
            "Enter your ingredients (comma-separated):",
            placeholder="e.g., egg, onions, tomato",
            height=100,
            key="ingredients_input",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        st.write("")  # Spacer
        recipe_button = st.button("Get Recipe Suggestions", use_container_width=True)

    # Sample ingredients
    st.info(
        "**Try these examples:** egg, onions | chicken, rice | tomato, pasta | potato, onion"
    )

    if recipe_button and ingredients_input:
        with st.spinner("Cooking up recipe suggestions..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/api/get-recipe",
                    json={"ingredients": ingredients_input},
                    timeout=300,  # Longer timeout for LLM
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display recipe
                    st.markdown("### Recipe Suggestion")

                    # Show which method was used
                    if data["generated_by"] == "ollama":
                        st.success("Generated using Local LLM (Ollama)")
                    else:
                        st.info("Retrieved from Recipe Database")

                    # Fix: Replace newlines before using in HTML
                    recipe_html = data["recipe"].replace("\n", "<br>")

                    # Display recipe in a card - FIXED: don't use f-string with backslash
                    st.markdown(
                        f"""
                    <div class="recipe-card">
                        {recipe_html}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Additional tips
                    with st.expander("Cooking Tips"):
                        st.write("""
                        - Always prep ingredients before cooking
                        - Adjust spices according to your taste
                        - You can substitute ingredients based on availability
                        - Follow food safety guidelines
                        """)

                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to API. Please ensure the backend server is running."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "Request timeout. The LLM might be taking too long. Please try again."
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif recipe_button:
        st.warning("Please enter some ingredients.")


def main():
    """Main application"""

    # Header
    st.markdown(
        '<div class="main-header"> Name Matching & Recipe Chatbot System</div>',
        unsafe_allow_html=True,
    )

    # Check API health
    api_status = check_api_health()

    if api_status:
        st.success("Backend API is running")
    else:
        st.error("Backend API is not running. Please start the backend server first.")
        st.code("cd backend\npython api.py", language="bash")
        st.stop()

    # Sidebar for task selection
    st.sidebar.title("Select Task")
    st.sidebar.write("Choose which functionality you want to test:")

    task_choice = st.sidebar.radio(
        "Tasks:", ["Task 1: Name Matching", "Task 2: Recipe Chatbot"], index=0
    )

    st.sidebar.divider()

    # About section in sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    **Task 1:** Uses fuzzy matching and semantic similarity (BERT embeddings) to find similar names.
    
    **Task 2:** Suggests recipes based on your ingredients using either a local LLM (Ollama) or a curated recipe database.
    """)

    st.sidebar.divider()

    # API info
    st.sidebar.title(" API Info")
    st.sidebar.code(f"Backend: {API_BASE_URL}", language="text")

    # Display selected task
    st.divider()

    if task_choice == "Task 1: Name Matching":
        display_task1()
    else:
        display_task2()

    # Footer
    st.divider()
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with FastAPI & Streamlit | Srihari </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

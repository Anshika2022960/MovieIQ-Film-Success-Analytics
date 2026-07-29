# ============================================================
# MovieIQ – Film Success Analytics Dashboard
# ============================================================

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieIQ Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    /* Main dashboard background */
    .stApp {
        background-color: #DFF3FF;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #C6E9FF;
    }

    /* Main content area */
    [data-testid="stMainBlockContainer"] {
        background-color: #DFF3FF;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# 2. DASHBOARD TITLE
# ============================================================

st.title("🎬 MovieIQ – Film Success Analytics Dashboard")

st.markdown(
    """
    This dashboard analyses historical movie data to identify the financial,
    audience-related and genre-related factors associated with movie success.

    A movie is considered **successful** when its revenue is greater than
    its production budget.
    """
)

st.divider()

def extract_main_genre(value):

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    if not text:
        return "Unknown"

    if text.startswith("[") and text.endswith("]"):

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list) and len(parsed) > 0:

                first_item = parsed[0]

                if isinstance(first_item, dict):
                    return str(
                        first_item.get("name", "Unknown")
                    ).strip()

                return str(first_item).strip()

        except (ValueError, SyntaxError):
            pass

    if "|" in text:
        return text.split("|")[0].strip()

    if "," in text:
        return text.split(",")[0].strip()

    return text



# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare the movie dataset.

    Parameters
    ----------
    file_path : str
        Path to the movies CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered movie data.
    """

    data = pd.read_csv(file_path)

    # Standardize column names
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    required_columns = [
        "title",
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average",
        "genres"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )

    # Remove duplicate rows
    data = data.drop_duplicates().copy()

    # Convert numeric columns safely
    numeric_columns = [
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove rows with missing or invalid budget/revenue
    data = data.dropna(
        subset=["budget", "revenue"]
    )

    data = data[
        (data["budget"] > 0) &
        (data["revenue"] > 0)
    ].copy()

    # Fill missing numeric values with median
    for column in ["popularity", "runtime", "vote_average"]:
        data[column] = data[column].fillna(
            data[column].median()
        )

    # Handle missing title and genre
    data["title"] = data["title"].fillna("Unknown Title")
    data["genres"] = data["genres"].fillna("Unknown")

    # Create success variable
    data["success"] = (
        data["revenue"] > data["budget"]
    ).astype(int)

    data["success_status"] = data["success"].map(
        {
            1: "Successful",
            0: "Unsuccessful"
        }
    )

    # Create profit
    data["profit"] = (
        data["revenue"] - data["budget"]
    )

    # Create return on investment
    data["roi"] = (
        data["profit"] / data["budget"]
    ) * 100

    # Extract the main genre
    data["main_genre"] = data["genres"].apply(
        extract_main_genre
    )

    return data


# ============================================================
# 4. GENRE PROCESSING FUNCTION
# ============================================================

def extract_main_genre(value) -> str:
    """
    Extract the first/main genre from different possible formats.

    It supports:
    - Action|Adventure
    - Action, Adventure
    - JSON-like genre lists
    - Python-list-like strings
    """

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    if not text:
        return "Unknown"

    # Handle JSON/Python list-style genre values
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list) and len(parsed) > 0:

                first_item = parsed[0]

                if isinstance(first_item, dict):
                    return str(
                        first_item.get("name", "Unknown")
                    ).strip()

                return str(first_item).strip()

        except (ValueError, SyntaxError):
            pass

    # Handle pipe-separated values
    if "|" in text:
        return text.split("|")[0].strip()

    # Handle comma-separated values
    if "," in text:
        return text.split(",")[0].strip()

    return text


# ============================================================
# 5. LOCATE DATASET
# ============================================================

possible_paths = [
    Path("data/movies.csv"),
    Path("movies.csv")
]

dataset_path = None

for path in possible_paths:
    if path.exists():
        dataset_path = path
        break


# ============================================================
# 6. DATA UPLOAD OPTION
# ============================================================

st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload movies.csv",
    type=["csv"]
)

try:

    if uploaded_file is not None:

        raw_uploaded_data = pd.read_csv(uploaded_file)

        temporary_path = "uploaded_movies.csv"
        raw_uploaded_data.to_csv(
            temporary_path,
            index=False
        )

        df = load_data(temporary_path)

        st.sidebar.success(
            "Uploaded dataset loaded successfully."
        )

    elif dataset_path is not None:

        df = load_data(str(dataset_path))

        st.sidebar.success(
            f"Loaded: {dataset_path}"
        )

    else:

        st.error(
            """
            Dataset not found.

            Place `movies.csv` inside the `data` folder,
            or upload the file using the sidebar.
            """
        )

        st.stop()

except Exception as error:

    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ============================================================
# 7. SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")

available_genres = sorted(
    df["main_genre"]
    .dropna()
    .unique()
    .tolist()
)

selected_genres = st.sidebar.multiselect(
    "Select Genre",
    options=available_genres,
    default=available_genres
)

minimum_vote = st.sidebar.slider(
    "Minimum Vote Average",
    min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()),
    value=float(df["vote_average"].min()),
    step=0.1
)

success_options = [
    "All",
    "Successful",
    "Unsuccessful"
]

selected_success = st.sidebar.selectbox(
    "Movie Success Status",
    options=success_options
)

minimum_runtime = int(df["runtime"].min())
maximum_runtime = int(df["runtime"].max())

runtime_range = st.sidebar.slider(
    "Runtime Range in Minutes",
    min_value=minimum_runtime,
    max_value=maximum_runtime,
    value=(minimum_runtime, maximum_runtime)
)

minimum_budget = float(df["budget"].min())
maximum_budget = float(df["budget"].max())

budget_range = st.sidebar.slider(
    "Budget Range",
    min_value=minimum_budget,
    max_value=maximum_budget,
    value=(minimum_budget, maximum_budget)
)


# ============================================================
# 8. APPLY FILTERS
# ============================================================

filtered_df = df[
    df["main_genre"].isin(selected_genres)
].copy()

filtered_df = filtered_df[
    filtered_df["vote_average"] >= minimum_vote
]

filtered_df = filtered_df[
    filtered_df["runtime"].between(
        runtime_range[0],
        runtime_range[1]
    )
]

filtered_df = filtered_df[
    filtered_df["budget"].between(
        budget_range[0],
        budget_range[1]
    )
]

if selected_success != "All":
    filtered_df = filtered_df[
        filtered_df["success_status"] == selected_success
    ]


# ============================================================
# 9. FILTERED DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        """
        No movies match the selected filters.

        Change the genre, rating, runtime or budget settings.
        """
    )

    st.stop()


# ============================================================
# 10. HELPER FUNCTION FOR CURRENCY
# ============================================================

def format_currency(value: float) -> str:
    """Format large currency values in readable form."""

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"

    if abs(value) >= 1_000:
        return f"${value / 1_000:,.2f}K"

    return f"${value:,.2f}"


# ============================================================
# 11. DASHBOARD NAVIGATION
# ============================================================

dashboard_page = st.sidebar.radio(
    "Select Dashboard Section",
    options=[
        "Overview",
        "Financial Analysis",
        "Genre Analysis",
        "Audience Analysis",
        "Statistical Testing",
        "Movie Explorer",
        "Business Insights"
    ]
)


# ============================================================
# 12. OVERVIEW PAGE
# ============================================================

if dashboard_page == "Overview":

    st.header("📊 Dashboard Overview")

    total_movies = len(filtered_df)

    successful_movies = int(
        filtered_df["success"].sum()
    )

    unsuccessful_movies = (
        total_movies - successful_movies
    )

    success_rate = (
        filtered_df["success"].mean() * 100
    )

    average_budget = filtered_df["budget"].mean()
    average_revenue = filtered_df["revenue"].mean()
    total_profit = filtered_df["profit"].sum()

    first_row = st.columns(4)

    first_row[0].metric(
        "Total Movies",
        f"{total_movies:,}"
    )

    first_row[1].metric(
        "Successful Movies",
        f"{successful_movies:,}"
    )

    first_row[2].metric(
        "Unsuccessful Movies",
        f"{unsuccessful_movies:,}"
    )

    first_row[3].metric(
        "Success Rate",
        f"{success_rate:.2f}%"
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "Average Budget",
        format_currency(average_budget)
    )

    second_row[1].metric(
        "Average Revenue",
        format_currency(average_revenue)
    )

    second_row[2].metric(
        "Total Profit",
        format_currency(total_profit)
    )

    st.divider()

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:

        st.subheader("Successful vs Unsuccessful Movies")

        fig, ax = plt.subplots(figsize=(7, 5))

        order = [
            "Successful",
            "Unsuccessful"
        ]

        sns.countplot(
            data=filtered_df,
            x="success_status",
            order=order,
            ax=ax
        )

        ax.set_xlabel("Movie Status")
        ax.set_ylabel("Number of Movies")
        ax.set_title(
            "Movie Success Distribution"
        )

        st.pyplot(fig)
        plt.close(fig)

    with chart_column_2:

        st.subheader("Vote Average Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.histplot(
            data=filtered_df,
            x="vote_average",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Vote Average")
        ax.set_ylabel("Number of Movies")
        ax.set_title(
            "Distribution of Movie Ratings"
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Filtered Dataset Preview")

    preview_columns = [
        "title",
        "main_genre",
        "budget",
        "revenue",
        "profit",
        "roi",
        "vote_average",
        "success_status"
    ]

    st.dataframe(
        filtered_df[preview_columns].head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 13. FINANCIAL ANALYSIS PAGE
# ============================================================

elif dashboard_page == "Financial Analysis":

    st.header("💰 Financial Analysis")

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:

        st.subheader("Budget vs Revenue")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=filtered_df,
            x="budget",
            y="revenue",
            hue="success_status",
            alpha=0.75,
            ax=ax
        )

        maximum_value = max(
            filtered_df["budget"].max(),
            filtered_df["revenue"].max()
        )

        ax.plot(
            [0, maximum_value],
            [0, maximum_value],
            linestyle="--",
            label="Revenue = Budget"
        )

        ax.set_xlabel("Budget")
        ax.set_ylabel("Revenue")
        ax.set_title("Budget vs Revenue")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            """
            Points above the diagonal line earned more revenue
            than their production budget.
            """
        )

    with chart_column_2:

        st.subheader("Profit Distribution")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.histplot(
            data=filtered_df,
            x="profit",
            bins=30,
            kde=True,
            ax=ax
        )

        ax.axvline(
            0,
            linestyle="--"
        )

        ax.set_xlabel("Profit")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Distribution of Movie Profit")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Top 10 Most Profitable Movies")

    top_profit_movies = (
        filtered_df[
            [
                "title",
                "main_genre",
                "budget",
                "revenue",
                "profit",
                "roi"
            ]
        ]
        .sort_values(
            by="profit",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_profit_movies,
        use_container_width=True,
        hide_index=True,
        column_config={
            "budget": st.column_config.NumberColumn(
                "Budget",
                format="$%.2f"
            ),
            "revenue": st.column_config.NumberColumn(
                "Revenue",
                format="$%.2f"
            ),
            "profit": st.column_config.NumberColumn(
                "Profit",
                format="$%.2f"
            ),
            "roi": st.column_config.NumberColumn(
                "ROI",
                format="%.2f%%"
            )
        }
    )

    st.subheader("Top 10 Movies by ROI")

    top_roi_movies = (
        filtered_df[
            [
                "title",
                "main_genre",
                "budget",
                "revenue",
                "profit",
                "roi"
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=["roi"]
        )
        .sort_values(
            by="roi",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_roi_movies,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 14. GENRE ANALYSIS PAGE
# ============================================================

elif dashboard_page == "Genre Analysis":

    st.header("🎭 Genre Analysis")

    genre_summary = (
        filtered_df
        .groupby("main_genre")
        .agg(
            movie_count=("title", "count"),
            successful_movies=("success", "sum"),
            success_rate=("success", "mean"),
            average_budget=("budget", "mean"),
            average_revenue=("revenue", "mean"),
            average_profit=("profit", "mean"),
            average_roi=("roi", "mean")
        )
        .reset_index()
    )

    genre_summary["success_rate"] = (
        genre_summary["success_rate"] * 100
    )

    genre_summary = genre_summary[
        genre_summary["movie_count"] >= 2
    ].copy()

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:

        st.subheader("Most Common Genres")

        common_genres = (
            genre_summary
            .sort_values(
                by="movie_count",
                ascending=False
            )
            .head(12)
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.barplot(
            data=common_genres,
            x="movie_count",
            y="main_genre",
            ax=ax
        )

        ax.set_xlabel("Number of Movies")
        ax.set_ylabel("Genre")
        ax.set_title("Most Common Movie Genres")

        st.pyplot(fig)
        plt.close(fig)

    with chart_column_2:

        st.subheader("Genre-wise Success Rate")

        successful_genres = (
            genre_summary
            .sort_values(
                by="success_rate",
                ascending=False
            )
            .head(12)
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.barplot(
            data=successful_genres,
            x="success_rate",
            y="main_genre",
            ax=ax
        )

        ax.set_xlabel("Success Rate (%)")
        ax.set_ylabel("Genre")
        ax.set_title("Movie Success Rate by Genre")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Average Revenue by Genre")

    revenue_by_genre = (
        genre_summary
        .sort_values(
            by="average_revenue",
            ascending=False
        )
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=revenue_by_genre,
        x="average_revenue",
        y="main_genre",
        ax=ax
    )

    ax.set_xlabel("Average Revenue")
    ax.set_ylabel("Genre")
    ax.set_title("Average Movie Revenue by Genre")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Genre Summary Table")

    st.dataframe(
        genre_summary.sort_values(
            by="success_rate",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "success_rate": st.column_config.NumberColumn(
                "Success Rate",
                format="%.2f%%"
            ),
            "average_roi": st.column_config.NumberColumn(
                "Average ROI",
                format="%.2f%%"
            )
        }
    )


# ============================================================
# 15. AUDIENCE ANALYSIS PAGE
# ============================================================

elif dashboard_page == "Audience Analysis":

    st.header("👥 Audience and Popularity Analysis")

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:

        st.subheader("Popularity by Success Status")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(
            data=filtered_df,
            x="success_status",
            y="popularity",
            ax=ax
        )

        ax.set_xlabel("Movie Status")
        ax.set_ylabel("Popularity")
        ax.set_title(
            "Popularity of Successful and Unsuccessful Movies"
        )

        st.pyplot(fig)
        plt.close(fig)

    with chart_column_2:

        st.subheader("Vote Average by Success Status")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(
            data=filtered_df,
            x="success_status",
            y="vote_average",
            ax=ax
        )

        ax.set_xlabel("Movie Status")
        ax.set_ylabel("Vote Average")
        ax.set_title(
            "Ratings of Successful and Unsuccessful Movies"
        )

        st.pyplot(fig)
        plt.close(fig)

    chart_column_3, chart_column_4 = st.columns(2)

    with chart_column_3:

        st.subheader("Runtime by Success Status")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.boxplot(
            data=filtered_df,
            x="success_status",
            y="runtime",
            ax=ax
        )

        ax.set_xlabel("Movie Status")
        ax.set_ylabel("Runtime in Minutes")
        ax.set_title(
            "Runtime by Movie Success"
        )

        st.pyplot(fig)
        plt.close(fig)

    with chart_column_4:

        st.subheader("Popularity vs Revenue")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=filtered_df,
            x="popularity",
            y="revenue",
            hue="success_status",
            alpha=0.75,
            ax=ax
        )

        ax.set_xlabel("Popularity")
        ax.set_ylabel("Revenue")
        ax.set_title(
            "Popularity vs Movie Revenue"
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")

    numeric_columns = [
        "budget",
        "revenue",
        "profit",
        "roi",
        "popularity",
        "runtime",
        "vote_average",
        "success"
    ]

    correlation_matrix = (
        filtered_df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .corr()
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numeric Movie Features"
    )

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 16. STATISTICAL TESTING PAGE
# ============================================================

elif dashboard_page == "Statistical Testing":

    st.header("🧪 Statistical Hypothesis Testing")

    st.info(
        """
        The tests below use the complete cleaned dataset rather than
        only the currently filtered subset. This gives more stable and
        representative statistical results.
        """
    )

    # --------------------------------------------------------
    # T-Test
    # --------------------------------------------------------

    st.subheader(
        "1. T-Test: Popularity and Movie Success"
    )

    st.markdown(
        """
        **Null hypothesis (H₀):** The mean popularity of successful
        and unsuccessful movies is the same.

        **Alternative hypothesis (H₁):** The mean popularity differs
        between successful and unsuccessful movies.
        """
    )

    successful_popularity = df.loc[
        df["success"] == 1,
        "popularity"
    ].dropna()

    unsuccessful_popularity = df.loc[
        df["success"] == 0,
        "popularity"
    ].dropna()

    if (
        len(successful_popularity) >= 2 and
        len(unsuccessful_popularity) >= 2
    ):

        t_statistic, t_p_value = ttest_ind(
            successful_popularity,
            unsuccessful_popularity,
            equal_var=False,
            nan_policy="omit"
        )

        t_col_1, t_col_2, t_col_3 = st.columns(3)

        t_col_1.metric(
            "T-statistic",
            f"{t_statistic:.4f}"
        )

        t_col_2.metric(
            "P-value",
            f"{t_p_value:.6f}"
        )

        t_col_3.metric(
            "Significance Level",
            "0.05"
        )

        successful_mean = (
            successful_popularity.mean()
        )

        unsuccessful_mean = (
            unsuccessful_popularity.mean()
        )

        st.write(
            f"Mean popularity of successful movies: "
            f"**{successful_mean:.2f}**"
        )

        st.write(
            f"Mean popularity of unsuccessful movies: "
            f"**{unsuccessful_mean:.2f}**"
        )

        if t_p_value < 0.05:

            st.success(
                """
                Reject the null hypothesis.

                The mean popularity differs significantly between
                successful and unsuccessful movies.
                """
            )

        else:

            st.warning(
                """
                Fail to reject the null hypothesis.

                The analysis does not provide sufficient evidence
                of a significant difference in mean popularity.
                """
            )

    else:

        st.warning(
            "There are not enough observations to perform the T-Test."
        )

    st.divider()

    # --------------------------------------------------------
    # Chi-Square Test
    # --------------------------------------------------------

    st.subheader(
        "2. Chi-Square Test: Genre and Movie Success"
    )

    st.markdown(
        """
        **Null hypothesis (H₀):** Genre and movie success are independent.

        **Alternative hypothesis (H₁):** Genre and movie success are
        associated.
        """
    )

    contingency_table = pd.crosstab(
        df["main_genre"],
        df["success_status"]
    )

    # Retain genres having sufficient observations
    valid_genres = contingency_table[
        contingency_table.sum(axis=1) >= 5
    ]

    if (
        valid_genres.shape[0] >= 2 and
        valid_genres.shape[1] >= 2
    ):

        (
            chi_square_statistic,
            chi_p_value,
            degrees_of_freedom,
            expected_frequencies
        ) = chi2_contingency(valid_genres)

        chi_col_1, chi_col_2, chi_col_3 = st.columns(3)

        chi_col_1.metric(
            "Chi-Square Statistic",
            f"{chi_square_statistic:.4f}"
        )

        chi_col_2.metric(
            "P-value",
            f"{chi_p_value:.6f}"
        )

        chi_col_3.metric(
            "Degrees of Freedom",
            f"{degrees_of_freedom}"
        )

        if chi_p_value < 0.05:

            st.success(
                """
                Reject the null hypothesis.

                Genre and movie success have a statistically
                significant association.
                """
            )

        else:

            st.warning(
                """
                Fail to reject the null hypothesis.

                The analysis does not provide sufficient evidence
                that genre and movie success are associated.
                """
            )

        with st.expander(
            "View Genre–Success Contingency Table"
        ):

            st.dataframe(
                valid_genres,
                use_container_width=True
            )

        expected_df = pd.DataFrame(
            expected_frequencies,
            index=valid_genres.index,
            columns=valid_genres.columns
        )

        low_expected_cells = (
            expected_df < 5
        ).sum().sum()

        total_expected_cells = expected_df.size

        low_expected_percentage = (
            low_expected_cells /
            total_expected_cells
        ) * 100

        st.caption(
            f"{low_expected_percentage:.2f}% of expected-frequency "
            "cells are below 5."
        )

        if low_expected_percentage > 20:

            st.warning(
                """
                More than 20% of expected frequencies are below 5.
                Interpret the Chi-Square result cautiously. You may
                combine rare genres into an 'Other' category.
                """
            )

    else:

        st.warning(
            """
            There are not enough genre categories or success classes
            to perform the Chi-Square test.
            """
        )

    st.divider()

    st.subheader("Understanding the P-value")

    st.markdown(
        """
        - When **p < 0.05**, reject the null hypothesis.
        - When **p ≥ 0.05**, fail to reject the null hypothesis.
        - A small p-value means the observed result would be unusual
          if the null hypothesis were true.
        - Statistical significance does not automatically mean that
          the relationship is practically large or economically important.
        """
    )


# ============================================================
# 17. MOVIE EXPLORER PAGE
# ============================================================

elif dashboard_page == "Movie Explorer":

    st.header("🔍 Movie Data Explorer")

    search_title = st.text_input(
        "Search by Movie Title"
    )

    displayed_df = filtered_df.copy()

    if search_title:

        displayed_df = displayed_df[
            displayed_df["title"]
            .str.contains(
                search_title,
                case=False,
                na=False
            )
        ]

    sort_column = st.selectbox(
        "Sort Movies By",
        options=[
            "revenue",
            "profit",
            "roi",
            "budget",
            "popularity",
            "vote_average",
            "runtime"
        ]
    )

    sort_order = st.radio(
        "Sort Order",
        options=[
            "Descending",
            "Ascending"
        ],
        horizontal=True
    )

    displayed_df = displayed_df.sort_values(
        by=sort_column,
        ascending=(sort_order == "Ascending")
    )

    columns_to_show = [
        "title",
        "main_genre",
        "budget",
        "revenue",
        "profit",
        "roi",
        "popularity",
        "runtime",
        "vote_average",
        "success_status"
    ]

    st.write(
        f"Displaying **{len(displayed_df):,}** movies."
    )

    st.dataframe(
        displayed_df[columns_to_show],
        use_container_width=True,
        hide_index=True
    )

    csv_data = displayed_df[
        columns_to_show
    ].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="movieiq_filtered_data.csv",
        mime="text/csv"
    )


# ============================================================
# 18. BUSINESS INSIGHTS PAGE
# ============================================================

elif dashboard_page == "Business Insights":

    st.header("💡 Data-Driven Business Insights")

    full_success_rate = df["success"].mean() * 100

    average_success_popularity = df.loc[
        df["success"] == 1,
        "popularity"
    ].mean()

    average_failure_popularity = df.loc[
        df["success"] == 0,
        "popularity"
    ].mean()

    genre_business_summary = (
        df.groupby("main_genre")
        .agg(
            movie_count=("title", "count"),
            success_rate=("success", "mean"),
            average_profit=("profit", "mean"),
            average_revenue=("revenue", "mean")
        )
        .reset_index()
    )

    genre_business_summary["success_rate"] *= 100

    reliable_genres = genre_business_summary[
        genre_business_summary["movie_count"] >= 5
    ].copy()

    if not reliable_genres.empty:

        best_success_genre = (
            reliable_genres
            .sort_values(
                by="success_rate",
                ascending=False
            )
            .iloc[0]
        )

        best_profit_genre = (
            reliable_genres
            .sort_values(
                by="average_profit",
                ascending=False
            )
            .iloc[0]
        )

        st.markdown(
            f"""
            ### Overall Performance

            - The dataset contains **{len(df):,} cleaned movies**.
            - The overall historical success rate is
              **{full_success_rate:.2f}%**.
            - Successful movies have an average popularity of
              **{average_success_popularity:.2f}**.
            - Unsuccessful movies have an average popularity of
              **{average_failure_popularity:.2f}**.

            ### Genre Findings

            - Among genres with at least five movies,
              **{best_success_genre['main_genre']}** has the highest
              historical success rate at
              **{best_success_genre['success_rate']:.2f}%**.
            - **{best_profit_genre['main_genre']}** has the highest
              average profit among sufficiently represented genres.
            """
        )

    else:

        st.info(
            """
            There are not enough movies per genre to produce reliable
            genre-level business insights.
            """
        )

    st.subheader("Recommendations")

    st.markdown(
        """
        1. Compare genre success rates together with the number of films
           in each genre. A high rate based on very few movies may be
           unreliable.

        2. Use both profit and ROI. A movie may earn a large absolute
           profit but still have a moderate return relative to its budget.

        3. Examine popularity and audience rating as supporting indicators,
           but do not assume that correlation proves causation.

        4. Review budget ranges before making investment decisions because
           a higher budget does not guarantee commercial success.

        5. Use MovieIQ as a historical decision-support dashboard rather
           than as a guarantee of future film performance.
        """
    )

    st.subheader("Project Limitation")

    st.warning(
        """
        The dashboard defines success only as revenue greater than budget.
        It does not account for marketing costs, distribution fees, taxes,
        theatre revenue sharing or inflation. Therefore, the calculated
        success status is a simplified measure of profitability.
        """
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.caption(
    """
    MovieIQ – Film Success Analytics Dashboard |
    Built using Python, Pandas, Seaborn, SciPy and Streamlit
    """
)
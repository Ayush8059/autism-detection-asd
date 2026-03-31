import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Tuple, List, Optional, Dict, Any
import io
import datetime

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part 1: Activity Database
# ---------------------------------------------------------------------------

def create_activity_database() -> pd.DataFrame:
    """
    Creates a comprehensive DataFrame of therapeutic activities for autism support.

    Returns:
        pd.DataFrame with columns: activity_id, name, skills_targeted,
        age_range, min_age, max_age, description, duration_minutes, difficulty
    """
    data = {
        'activity_id': list(range(101, 131)),
        'name': [
            "Sensory Rice Bin",
            "Play-Doh Sculpting",
            "Swinging",
            "Threading Beads",
            "Simple Board Game",
            "Social Story Time",
            "Bubble Blowing",
            "Trampoline Jumping",
            "Picture Exchange (PECS)",
            "Building with LEGOs",
            "Listening to Calming Music",
            "Joint Drawing",
            "Weighted Blanket Rest",
            "Obstacle Course",
            "Sand Tray Play",
            "Mirror Exercises",
            "Sorting & Categorising",
            "Finger Painting",
            "Balance Board",
            "Puppet Play",
            "Emotion Flashcards",
            "Simon Says",
            "Deep Pressure Massage",
            "Jigsaw Puzzles",
            "Cooking Simple Recipes",
            "Nature Walk & Scavenger Hunt",
            "Rhythm & Drumming",
            "Yoga for Kids",
            "Story Sequencing Cards",
            "Coin & Button Sorting",
        ],
        'skills_targeted': [
            "sensory_integration fine_motor calming",
            "fine_motor sensory_integration creative",
            "sensory_integration gross_motor vestibular calming",
            "fine_motor concentration hand_eye_coordination",
            "social_skills turn_taking rules_following",
            "social_skills communication comprehension emotional_regulation",
            "communication oral_motor joint_attention",
            "gross_motor sensory_integration proprioceptive energy_release",
            "communication non_verbal requesting choice_making",
            "fine_motor creative problem_solving concentration",
            "calming sensory_integration emotional_regulation auditory",
            "social_skills joint_attention communication creative",
            "calming sensory_integration emotional_regulation proprioceptive",
            "gross_motor vestibular proprioceptive energy_release coordination",
            "sensory_integration creative calming fine_motor",
            "social_skills joint_attention non_verbal communication",
            "concentration problem_solving fine_motor hand_eye_coordination",
            "fine_motor creative sensory_integration emotional_regulation",
            "vestibular gross_motor proprioceptive concentration",
            "social_skills communication creative emotional_regulation",
            "emotional_regulation social_skills comprehension communication",
            "gross_motor social_skills rules_following turn_taking auditory",
            "calming sensory_integration proprioceptive emotional_regulation",
            "problem_solving concentration fine_motor hand_eye_coordination",
            "fine_motor concentration social_skills choice_making",
            "gross_motor sensory_integration joint_attention concentration",
            "auditory sensory_integration emotional_regulation gross_motor",
            "gross_motor calming proprioceptive vestibular emotional_regulation",
            "comprehension communication problem_solving concentration",
            "fine_motor concentration hand_eye_coordination problem_solving",
        ],
        'age_range': [
            "3-5", "3-8", "3-10", "4-7", "5-10", "4-10", "3-6", "4-10",
            "3-8", "5-10", "3-12", "4-9", "3-12", "4-10", "3-8", "4-10",
            "3-7", "3-6", "5-10", "4-9", "4-10", "4-10", "3-10", "5-12",
            "6-12", "4-12", "4-10", "4-10", "5-10", "4-8",
        ],
        'description': [
            "Hiding toys in coloured rice encourages tactile exploration and focus.",
            "Moulding and shaping clay builds finger strength and imagination.",
            "Rhythmic swinging provides calming vestibular input.",
            "Threading coloured beads onto string improves hand-eye coordination.",
            "Turn-based games introduce rule-following and social interaction.",
            "Reading social stories together builds empathy and comprehension.",
            "Blowing bubbles targets breath control and joint attention.",
            "Bouncing on a trampoline releases energy and improves body awareness.",
            "Using picture cards to communicate requests and make choices.",
            "Structured building play supports spatial reasoning and focus.",
            "Carefully chosen music reduces anxiety and supports regulation.",
            "Side-by-side drawing fosters shared attention and creativity.",
            "Wrapping in a weighted blanket provides deep pressure and calm.",
            "Moving through a physical obstacle course builds coordination.",
            "Open-ended sand play supports sensory processing and creativity.",
            "Imitating facial expressions in a mirror builds social awareness.",
            "Grouping objects by colour, shape or size builds logic skills.",
            "Free-form painting encourages expression and sensory exploration.",
            "Standing on a wobble board improves balance and core strength.",
            "Puppet-based role play develops communication and storytelling.",
            "Identifying emotions on cards builds emotional literacy.",
            "Following spoken instructions combines listening and movement.",
            "Firm tactile pressure applied to limbs promotes relaxation.",
            "Completing jigsaws builds visual perception and persistence.",
            "Following simple recipes develops sequencing and fine motor skills.",
            "Outdoor exploration supports attention, movement and curiosity.",
            "Playing percussion instruments develops rhythm and regulation.",
            "Child-friendly yoga poses build body awareness and calm.",
            "Arranging picture cards in order supports narrative comprehension.",
            "Picking up small objects improves pincer grip and focus.",
        ],
        'duration_minutes': [
            20, 20, 15, 20, 30, 20, 10, 20,
            15, 30, 20, 20, 15, 20, 20, 15,
            20, 20, 15, 20, 20, 20, 15, 30,
            40, 30, 20, 20, 20, 15,
        ],
        'difficulty': [
            "Easy", "Easy", "Easy", "Medium", "Medium", "Easy", "Easy", "Easy",
            "Easy", "Medium", "Easy", "Easy", "Easy", "Medium", "Easy", "Medium",
            "Easy", "Easy", "Medium", "Medium", "Medium", "Easy", "Easy", "Medium",
            "Hard", "Medium", "Medium", "Medium", "Medium", "Easy",
        ],
    }

    # Parse numeric age bounds for filtering
    min_ages, max_ages = [], []
    for ar in data['age_range']:
        lo, hi = ar.split('-')
        min_ages.append(int(lo))
        max_ages.append(int(hi))
    data['min_age'] = min_ages
    data['max_age'] = max_ages

    df = pd.DataFrame(data)
    if df.isnull().any().any():
        logger.warning("Missing values detected in activity database")
    logger.info(f"Activity database created with {len(df)} activities")
    return df


# ---------------------------------------------------------------------------
# Part 2: Age Filtering
# ---------------------------------------------------------------------------

def filter_activities_by_age(df: pd.DataFrame, child_age: int) -> pd.DataFrame:
    """
    Returns only activities whose age range includes child_age.

    Args:
        df:         Full activity DataFrame (from create_activity_database)
        child_age:  Child's age in years (integer, 1–18)

    Returns:
        Filtered DataFrame; original DataFrame unchanged.
    """
    if child_age < 1 or child_age > 18:
        logger.warning(f"Unusual child age provided: {child_age}")
    filtered = df[(df['min_age'] <= child_age) & (df['max_age'] >= child_age)].copy()
    logger.info(f"Age filter ({child_age} yrs): {len(filtered)}/{len(df)} activities retained")
    return filtered


# ---------------------------------------------------------------------------
# Part 3: Activity Recommender
# ---------------------------------------------------------------------------

class ActivityRecommender:
    """
    TF-IDF + cosine-similarity recommender for autism support activities.
    Accepts an optional pre-filtered DataFrame so age filtering is applied
    before similarity scoring.
    """

    def __init__(self, activities_df: pd.DataFrame):
        if 'skills_targeted' not in activities_df.columns:
            raise ValueError("DataFrame must contain 'skills_targeted' column")
        self.activities_df = activities_df
        self._fit(activities_df)

    def _fit(self, df: pd.DataFrame):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['skills_targeted'])
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def recommend(
        self,
        child_profile_needs: str,
        num_recommendations: int = 5,
        age_filtered_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Recommend activities for the given needs string.

        Args:
            child_profile_needs:  Space-separated skill names.
            num_recommendations:  Max results to return.
            age_filtered_df:      If provided, score only these rows.

        Returns:
            DataFrame of top activities with a 'similarity_score' column.
        """
        if not child_profile_needs or not child_profile_needs.strip():
            return pd.DataFrame()
        if num_recommendations <= 0:
            return pd.DataFrame()

        try:
            if age_filtered_df is not None and not age_filtered_df.empty:
                # Re-fit vectorizer on the filtered subset for accurate scoring
                local_vec = TfidfVectorizer(
                    stop_words='english', lowercase=True,
                    analyzer='word', token_pattern=r'(?u)\b\w+\b',
                )
                local_matrix = local_vec.fit_transform(age_filtered_df['skills_targeted'])
                profile_vec = local_vec.transform([child_profile_needs.strip()])
                sims = cosine_similarity(profile_vec, local_matrix).flatten()
                source_df = age_filtered_df
            else:
                profile_vec = self.tfidf_vectorizer.transform([child_profile_needs.strip()])
                sims = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()
                source_df = self.activities_df

            sorted_idx = sims.argsort()[::-1]
            top_idx = [i for i in sorted_idx if sims[i] > 0][:num_recommendations]

            if not top_idx:
                return pd.DataFrame()

            recs = source_df.iloc[top_idx].copy()
            recs['similarity_score'] = sims[top_idx]
            logger.info(f"Generated {len(recs)} recommendations")
            return recs

        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Part 4: Confidence Tier
# ---------------------------------------------------------------------------

def get_confidence_tier(raw_score: float, label: str) -> Dict[str, str]:
    """
    Converts a raw sigmoid score into a human-readable confidence tier
    with an associated colour for UI display.

    Args:
        raw_score:  Sigmoid output from the model (0.0 – 1.0).
        label:      "Autistic" or "Non_Autistic".

    Returns:
        Dict with keys: tier (str), color (str), description (str).
        color is a hex string suitable for st.markdown / reportlab.
    """
    # Distance from decision boundary (0.5) gives certainty
    certainty = abs(raw_score - 0.5) * 2  # 0.0 = boundary, 1.0 = max

    if certainty >= 0.70:
        tier = "High Confidence"
        color = "#1a6e3c"       # dark green
        bg_color = "#d4edda"
    elif certainty >= 0.35:
        tier = "Moderate Confidence"
        color = "#856404"       # amber
        bg_color = "#fff3cd"
    else:
        tier = "Low Confidence"
        color = "#721c24"       # red
        bg_color = "#f8d7da"

    if label == "Autistic":
        confidence_pct = (1.0 - raw_score) * 100
    else:
        confidence_pct = raw_score * 100

    description = (
        f"{tier} — the model assigned {confidence_pct:.1f}% probability "
        f"to the '{label}' classification."
    )

    return {
        "tier": tier,
        "color": color,
        "bg_color": bg_color,
        "confidence_pct": round(confidence_pct, 2),
        "description": description,
    }


# ---------------------------------------------------------------------------
# Part 5: PDF Report Generation
# ---------------------------------------------------------------------------

def generate_pdf_report(
    prediction_label: str,
    raw_score: float,
    child_age: Optional[int],
    selected_needs: List[str],
    recommendations: pd.DataFrame,
    timestamp: Optional[datetime.datetime] = None,
) -> bytes:
    """
    Generates a professional clinical-style PDF report as bytes (suitable
    for st.download_button).

    Args:
        prediction_label:  "Autistic" or "Non_Autistic"
        raw_score:         Raw sigmoid output (0.0 – 1.0)
        child_age:         Child's age in years, or None
        selected_needs:    List of selected developmental need strings
        recommendations:   DataFrame returned by ActivityRecommender.recommend()
        timestamp:         Datetime of analysis; defaults to now

    Returns:
        PDF file content as bytes.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Autism Support System — Assessment Report",
        author="Autism Support System",
    )

    # ── Colour palette (clinical blue-grey) ──────────────────────────────
    NAVY      = colors.HexColor("#1B2A4A")
    STEEL     = colors.HexColor("#4A6FA5")
    LIGHT     = colors.HexColor("#EEF2F7")
    MID_GREY  = colors.HexColor("#6C757D")
    WHITE     = colors.white
    GREEN     = colors.HexColor("#1a6e3c")
    AMBER     = colors.HexColor("#856404")
    RED_C     = colors.HexColor("#721c24")

    # ── Styles ─────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, **kw)

    S_TITLE = style("ReportTitle",
        fontSize=22, leading=28, textColor=NAVY,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
    S_SUBTITLE = style("Subtitle",
        fontSize=10, textColor=MID_GREY,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=16)
    S_SECTION = style("Section",
        fontSize=12, leading=16, textColor=NAVY,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4)
    S_BODY = style("Body",
        fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
        fontName="Helvetica", spaceAfter=6)
    S_SMALL = style("Small",
        fontSize=8, leading=11, textColor=MID_GREY,
        fontName="Helvetica", spaceAfter=4)
    S_DISCLAIMER = style("Disclaimer",
        fontSize=8, leading=11, textColor=MID_GREY,
        fontName="Helvetica-Oblique", alignment=TA_CENTER)

    conf_info = get_confidence_tier(raw_score, prediction_label)

    # ── Story ──────────────────────────────────────────────────────────────
    story = []

    # Header block
    story.append(Paragraph("Autism Support System", S_TITLE))
    story.append(Paragraph("Developmental Screening Report", S_SUBTITLE))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=10))

    # Meta table
    age_str = f"{child_age} years" if child_age else "Not specified"
    meta_data = [
        ["Report Generated", timestamp.strftime("%d %B %Y, %H:%M")],
        ["Child Age",         age_str],
        ["Model",             "VGG16 Fine-tuned (autism_detection_vgg16_finetuned.h5)"],
    ]
    meta_table = Table(meta_data, colWidths=[5 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR",   (1, 0), (1, -1), colors.HexColor("#333333")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Prediction Result ──────────────────────────────────────────────────
    story.append(Paragraph("1. Prediction Result", S_SECTION))
    story.append(HRFlowable(width="100%", thickness=0.5, color=STEEL, spaceAfter=8))

    pred_color = GREEN if prediction_label == "Non_Autistic" else AMBER
    pred_data = [
        [Paragraph("<b>Classification</b>", S_BODY),
         Paragraph(f"<b>{prediction_label.replace('_', ' ')}</b>", S_BODY)],
        [Paragraph("<b>Confidence</b>", S_BODY),
         Paragraph(f"{conf_info['confidence_pct']:.1f}%", S_BODY)],
        [Paragraph("<b>Confidence Tier</b>", S_BODY),
         Paragraph(conf_info['tier'], S_BODY)],
        [Paragraph("<b>Raw Model Score</b>", S_BODY),
         Paragraph(f"{raw_score:.4f}  (threshold: 0.5000)", S_BODY)],
    ]
    pred_table = Table(pred_data, colWidths=[5 * cm, 12 * cm])
    pred_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("TEXTCOLOR",   (1, 0), (1, 0), pred_color),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(conf_info['description'], S_SMALL))

    # ── Developmental Needs ────────────────────────────────────────────────
    story.append(Paragraph("2. Selected Developmental Needs", S_SECTION))
    story.append(HRFlowable(width="100%", thickness=0.5, color=STEEL, spaceAfter=8))

    if selected_needs:
        needs_text = ",  ".join(n.replace('_', ' ').title() for n in selected_needs)
        story.append(Paragraph(needs_text, S_BODY))
    else:
        story.append(Paragraph("No developmental needs were selected.", S_BODY))

    # ── Recommended Activities ─────────────────────────────────────────────
    story.append(Paragraph("3. Recommended Activities", S_SECTION))
    story.append(HRFlowable(width="100%", thickness=0.5, color=STEEL, spaceAfter=8))

    if recommendations is not None and not recommendations.empty:
        act_header = [
            Paragraph("<b>#</b>", S_BODY),
            Paragraph("<b>Activity</b>", S_BODY),
            Paragraph("<b>Age Range</b>", S_BODY),
            Paragraph("<b>Duration</b>", S_BODY),
            Paragraph("<b>Skills Targeted</b>", S_BODY),
        ]
        act_rows = [act_header]
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            skills_fmt = ", ".join(
                s.replace('_', ' ').title() for s in row['skills_targeted'].split()
            )
            dur = f"{row['duration_minutes']} min" if 'duration_minutes' in row else "—"
            act_rows.append([
                Paragraph(str(i), S_BODY),
                Paragraph(f"<b>{row['name']}</b>", S_BODY),
                Paragraph(row['age_range'], S_BODY),
                Paragraph(dur, S_BODY),
                Paragraph(skills_fmt, S_SMALL),
            ])

        act_table = Table(
            act_rows,
            colWidths=[0.8 * cm, 4.5 * cm, 2.2 * cm, 2 * cm, 7.5 * cm],
        )
        act_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, WHITE]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(KeepTogether(act_table))

        # Activity descriptions
        story.append(Spacer(1, 12))
        story.append(Paragraph("Activity Descriptions", S_SECTION))
        story.append(HRFlowable(width="100%", thickness=0.5, color=STEEL, spaceAfter=8))
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            if 'description' in row:
                story.append(Paragraph(f"<b>{i}. {row['name']}</b>", S_BODY))
                story.append(Paragraph(row['description'], S_BODY))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No activity recommendations were generated for this session.", S_BODY))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an automated screening tool and is "
        "intended for informational purposes only. It does not constitute a clinical "
        "diagnosis and should not replace assessment by a qualified healthcare "
        "professional. If you have concerns about a child's development, please "
        "consult a licensed clinician.",
        S_DISCLAIMER,
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    logger.info(f"PDF report generated ({len(pdf_bytes)} bytes)")
    return pdf_bytes


# ---------------------------------------------------------------------------
# Part 6: Image Prediction
# ---------------------------------------------------------------------------

def load_prediction_model(
    model_path: str = "autism_detection_vgg16_finetuned.h5",
) -> Optional[tf.keras.Model]:
    """Loads the trained Keras model; returns None on failure."""
    try:
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def preprocess_image(
    img_bytes: io.BytesIO,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Resizes and normalises an image to match training preprocessing
    (ImageDataGenerator rescale=1./255).
    """
    try:
        img_bytes.seek(0)
        img = image.load_img(img_bytes, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0          # matches training: rescale=1./255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None


def predict_from_image(
    image_bytes: io.BytesIO,
    model: tf.keras.Model,
) -> Tuple[str, float]:
    """
    Runs inference on an image.

    Keras flow_from_directory assigns class indices alphabetically:
        autistic     → 0  (sigmoid output < 0.5)
        non_autistic → 1  (sigmoid output > 0.5)

    Returns:
        (label, raw_sigmoid_score)
        raw_score close to 0.0 → Autistic
        raw_score close to 1.0 → Non_Autistic
    """
    if model is None:
        return "Error: Model not loaded.", 0.0
    try:
        if not isinstance(image_bytes, io.BytesIO):
            image_bytes = io.BytesIO(image_bytes)
        img_array = preprocess_image(image_bytes)
        if img_array is None:
            return "Error: Image preprocessing failed", 0.0

        prediction_value = float(model.predict(img_array, verbose=0)[0][0])
        label = "Non_Autistic" if prediction_value > 0.5 else "Autistic"
        logger.info(f"Prediction: {label} (raw={prediction_value:.4f})")
        return label, prediction_value

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0.0


# ---------------------------------------------------------------------------
# Part 7: Misc helpers
# ---------------------------------------------------------------------------

def get_all_available_skills() -> List[str]:
    """Returns a sorted list of all unique skill names in the activity database."""
    try:
        df = create_activity_database()
        skills = {s for row in df['skills_targeted'] for s in row.split() if s}
        return sorted(skills)
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []


def validate_model_compatibility(
    model: tf.keras.Model,
    input_shape: Tuple[int, ...] = (224, 224, 3),
) -> bool:
    """Returns True if the model's input shape matches input_shape."""
    try:
        if model.input_shape[1:] != input_shape:
            logger.warning(
                f"Shape mismatch: model={model.input_shape[1:]}, expected={input_shape}"
            )
            return False
        logger.info("Model compatibility check passed")
        return True
    except Exception as e:
        logger.error(f"Compatibility check error: {e}")
        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("TESTING HELPER MODULE")
    print("=" * 55)

    db = create_activity_database()
    print(f"\nActivity database: {len(db)} activities")

    # Age filter test
    filtered = filter_activities_by_age(db, 5)
    print(f"Age-filtered (5 yrs): {len(filtered)} activities")

    # Recommender test
    rec = ActivityRecommender(db)
    needs = "fine_motor calming sensory_integration"
    results = rec.recommend(needs, num_recommendations=3, age_filtered_df=filtered)
    print(f"\nTop recommendations for '{needs}':")
    for _, row in results.iterrows():
        print(f"  - {row['name']}  (score={row['similarity_score']:.3f})")

    # Confidence tier test
    for score, lbl in [(0.82, "Non_Autistic"), (0.38, "Autistic"), (0.52, "Non_Autistic")]:
        t = get_confidence_tier(score, lbl)
        print(f"\n  raw={score}  label={lbl}  →  {t['tier']}  {t['confidence_pct']}%")

    # PDF test
    pdf = generate_pdf_report(
        prediction_label="Autistic",
        raw_score=0.28,
        child_age=6,
        selected_needs=["fine_motor", "calming", "sensory_integration"],
        recommendations=results,
    )
    with open("/tmp/test_report.pdf", "wb") as f:
        f.write(pdf)
    print(f"\nPDF written to /tmp/test_report.pdf ({len(pdf)} bytes)")
    print("\nAll tests passed.")
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from datetime import datetime
from io import BytesIO
import fitz
import hashlib


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Floor Plan Analyzer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Floor Plan Analyzer")

st.write(
    "Upload a floor plan image or PDF to detect objects, "
    "view quantities and estimate preliminary costs."
)


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    weights = hf_hub_download(
        repo_id="mudasir13cs/floorcad-yolov8n-detect",
        filename="floorcad-yolov8n-detect.pt"
    )


    model = YOLO(weights)
    print("MODEL LABELS:", model.names)
    return model


def get_class_name(model, class_id):
    """
    Get a readable object name from the YOLO model.
    Unknown dataset labels are clearly marked.
    """
    names = model.names

    if isinstance(names, dict):
        name = names.get(class_id, f"class_{class_id}")
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        name = names[class_id]
    else:
        name = f"class_{class_id}"

    name = str(name)

    # Identify labels whose meanings are not documented.
    unknown_labels = ["class_31", "class_32", "class_34", "class_35"]

    if name in unknown_labels:
        return f"Unknown object ({name})"

    return name.replace("_", " ").title()

    """
    Safely get a class name from the YOLO model.
    Some models store names as a dictionary; others use a list.
    """
    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    return f"class_{class_id}"


# ==========================================
# 3. GET PDF PAGE COUNT
# ==========================================

def get_pdf_page_count(uploaded_file):
    pdf_document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    try:
        return len(pdf_document)
    finally:
        pdf_document.close()


# ==========================================
# 4. CONVERT PDF PAGE TO IMAGE
# ==========================================

def convert_to_image(uploaded_file, page_number=1):
    file_bytes = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".pdf"):
        pdf_document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        try:
            if page_number < 1 or page_number > len(pdf_document):
                raise ValueError("Invalid PDF page number.")

            page = pdf_document.load_page(page_number - 1)

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            return image

        finally:
            pdf_document.close()

    image = Image.open(BytesIO(file_bytes))
    return image.convert("RGB")


# ==========================================
# 5. SESSION STATE
# ==========================================

if "page_results" not in st.session_state:
    st.session_state.page_results = []

if "upload_hash" not in st.session_state:
    st.session_state.upload_hash = None

if "source_filename" not in st.session_state:
    st.session_state.source_filename = None

if "selected_pages" not in st.session_state:
    st.session_state.selected_pages = []


# ==========================================
# 6. FILE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your floor plan",
    type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file is not None:

    current_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    # Clear previous results when a new file is uploaded.
    if current_hash != st.session_state.upload_hash:
        st.session_state.page_results = []
        st.session_state.source_filename = None
        st.session_state.selected_pages = []
        st.session_state.upload_hash = current_hash

    st.success(f"Uploaded: {uploaded_file.name}")

    # ==========================================
    # 7. SELECT PDF PAGES
    # ==========================================

    selected_pages = []

    is_pdf = uploaded_file.name.lower().endswith(".pdf")

    if is_pdf:
        try:
            total_pages = get_pdf_page_count(uploaded_file)

            if total_pages == 0:
                st.error("This PDF contains no pages.")
                st.stop()

            st.info(f"Your PDF contains {total_pages} pages.")

            st.subheader("📄 Select Pages to Analyse")

            selection_mode = st.radio(
                "Choose how you want to analyse the PDF:",
                [
                    "All pages",
                    "Specific pages",
                    "Page range"
                ],
                horizontal=True
            )

            # OPTION 1: ALL PAGES
            if selection_mode == "All pages":
                selected_pages = list(range(1, total_pages + 1))

                st.success(
                    f"All {total_pages} pages selected."
                )

            # OPTION 2: SPECIFIC PAGES
            elif selection_mode == "Specific pages":
                selected_pages = st.multiselect(
                    "Choose individual pages:",
                    options=list(range(1, total_pages + 1)),
                    default=[1],
                    placeholder="Select PDF pages"
                )

                if selected_pages:
                    st.success(
                        f"Selected pages: "
                        f"{', '.join(map(str, sorted(selected_pages)))}"
                    )
                else:
                    st.warning("Please select at least one page.")

            # OPTION 3: PAGE RANGE
            elif selection_mode == "Page range":
                col1, col2 = st.columns(2)

                with col1:
                    start_page = st.number_input(
                        "Starting page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1
                    )

                with col2:
                    end_page = st.number_input(
                        "Ending page",
                        min_value=1,
                        max_value=total_pages,
                        value=total_pages,
                        step=1
                    )

                if start_page <= end_page:
                    selected_pages = list(
                        range(
                            int(start_page),
                            int(end_page) + 1
                        )
                    )

                    st.success(
                        f"Selected pages: {start_page} to {end_page}"
                    )
                else:
                    st.error(
                        "Ending page must be greater than or equal "
                        "to starting page."
                    )

        except Exception as e:
            st.error(f"Unable to read PDF: {e}")
            st.stop()

    else:
        selected_pages = [1]
        st.info("Image uploaded successfully.")

    # ==========================================
    # 8. ANALYSE SELECTED PAGES
    # ==========================================

    if st.button(
        "🔍 Analyse Selected Pages",
        type="primary"
    ):

        if not selected_pages:
            st.warning("Please select at least one page.")

        else:
            try:
                with st.spinner("Loading the YOLO model..."):
                    model = load_model()

                all_page_results = []

                progress = st.progress(0)
                status = st.empty()

                total_selected = len(selected_pages)

                for index, page_number in enumerate(selected_pages):
                    status.write(
                        f"Analysing page {page_number} "
                        f"({index + 1}/{total_selected})..."
                    )

                    # Convert selected page into an image.
                    image = convert_to_image(
                        uploaded_file,
                        page_number
                    )

                    # Run YOLO object detection.
                    results = model.predict(
                        source=image,
                        conf=0.35,
                        imgsz=640,
                        verbose=False
                    )

                    result = results[0]
                    detected_objects = []

                    # Read detected objects.
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            object_name = get_class_name(model, class_id)

                            detected_objects.append({
                                "Page": page_number,
                                "Object": object_name,
                                "Confidence": confidence * 100
                            })

                    # Create page-wise quantity report.
                    if detected_objects:
                        detection_df = pd.DataFrame(detected_objects)

                        page_df = detection_df.groupby(
                            "Object"
                        ).agg(
                            Quantity=("Object", "count"),
                            Average_Confidence=("Confidence", "mean")
                        ).reset_index()

                        page_df["Average_Confidence"] = (
                            page_df["Average_Confidence"].round(2)
                        )

                        page_df.insert(0, "Page", page_number)

                    else:
                        page_df = pd.DataFrame(columns=[
                            "Page",
                            "Object",
                            "Quantity",
                            "Average_Confidence"
                        ])

                    # Create annotated image.
                    annotated = result.plot()

                    # YOLO plot output is BGR; PIL expects RGB.
                    annotated_rgb = Image.fromarray(
                        annotated[:, :, ::-1].copy()
                    )

                    # Save page results.
                    all_page_results.append({
                        "page_number": page_number,
                        "original_image": image,
                        "annotated_image": annotated_rgb,
                        "quantity_report": page_df,
                        "detections": detected_objects
                    })

                    progress.progress(
                        (index + 1) / total_selected
                    )

                # Save all results in Streamlit session state.
                st.session_state.page_results = all_page_results
                st.session_state.source_filename = uploaded_file.name
                st.session_state.selected_pages = selected_pages

                status.success("Analysis completed successfully!")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.exception(e)


# ==========================================
# 9. DISPLAY ANALYSIS RESULTS
# ==========================================

if st.session_state.page_results:

    page_results = st.session_state.page_results

    st.divider()

    st.header("📊 Floor Plan Analysis Results")

    # Combine all page-wise quantity reports.
    page_reports = [
        result["quantity_report"]
        for result in page_results
        if not result["quantity_report"].empty
    ]

    if page_reports:
        pagewise_df = pd.concat(
            page_reports,
            ignore_index=True
        )
    else:
        pagewise_df = pd.DataFrame(columns=[
            "Page",
            "Object",
            "Quantity",
            "Average_Confidence"
        ])

    # Combine individual detections.
    all_detections = []

    for result in page_results:
        all_detections.extend(result["detections"])

    if all_detections:
        detailed_df = pd.DataFrame(all_detections)
    else:
        detailed_df = pd.DataFrame(columns=[
            "Page",
            "Object",
            "Confidence"
        ])

    # Overall quantity report.
    if not pagewise_df.empty:
        overall_df = pagewise_df.groupby(
            "Object"
        ).agg(
            Quantity=("Quantity", "sum"),
            Average_Confidence=("Average_Confidence", "mean")
        ).reset_index()

        overall_df["Average_Confidence"] = (
            overall_df["Average_Confidence"].round(2)
        )

    else:
        overall_df = pd.DataFrame(columns=[
            "Object",
            "Quantity",
            "Average_Confidence"
        ])

    # Summary metrics.
    total_objects = int(overall_df["Quantity"].sum())
    total_categories = len(overall_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages Analysed",
            len(page_results)
        )

    with col2:
        st.metric(
            "Total Objects Detected",
            total_objects
        )

    with col3:
        st.metric(
            "Object Categories",
            total_categories
        )

    # ==========================================
    # 10. PAGE-WISE RESULTS
    # ==========================================

    st.divider()

    st.header("📑 Page-wise Analysis")

    for result in page_results:
        page_number = result["page_number"]

        with st.expander(
            f"Page {page_number} - View Analysis",
            expanded=True
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Floor Plan")

                st.image(
                    result["original_image"],
                    use_container_width=True
                )

            with col2:
                st.subheader("Detected Objects")

                st.image(
                    result["annotated_image"],
                    use_container_width=True
                )

            st.subheader(
                f"Object Report - Page {page_number}"
            )

            if not result["quantity_report"].empty:
                st.dataframe(
                    result["quantity_report"].drop(
                        columns=["Page"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No objects detected on this page.")

    # ==========================================
    # 11. OVERALL QUANTITY REPORT
    # ==========================================

    st.divider()

    st.header("📋 Overall Quantity Report")

    st.write(
        "This report combines the detected quantities "
        "from all selected pages."
    )

    st.dataframe(
        overall_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # 12. PRELIMINARY COST ESTIMATOR
    # ==========================================

    st.divider()

    st.header("💰 Preliminary Cost Estimator")

    st.write(
        "Enter the unit rate for each object. "
        "Estimated amount = Total quantity × Unit rate."
    )

    st.warning(
        "This is a preliminary object-based estimate only. "
        "It is not a complete construction cost estimate. "
        "Verify detected quantities and rates before using them."
    )

    cost_data = []

    if not overall_df.empty:
        for _, row in overall_df.iterrows():
            object_name = row["Object"]
            quantity = int(row["Quantity"])
            confidence = float(row["Average_Confidence"])

            unit_rate = st.number_input(
                f"Rate per {object_name} (₹)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key=f"rate_{object_name}"
            )

            estimated_amount = quantity * unit_rate

            cost_data.append({
                "Object": object_name,
                "Quantity": quantity,
                "Average Confidence (%)": confidence,
                "Unit Rate (₹)": unit_rate,
                "Estimated Amount (₹)": estimated_amount,
                "Unit": "Per item"
            })

        cost_df = pd.DataFrame(cost_data)

        total_cost = cost_df["Estimated Amount (₹)"].sum()

        st.subheader("Combined Cost Estimate")

        st.dataframe(
            cost_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Preliminary Estimated Cost",
            f"₹{total_cost:,.2f}"
        )

    else:
        cost_df = pd.DataFrame(columns=[
            "Object",
            "Quantity",
            "Average Confidence (%)",
            "Unit Rate (₹)",
            "Estimated Amount (₹)",
            "Unit"
        ])

        total_cost = 0

        st.warning(
            "No objects were detected in the selected pages."
        )

    # ==========================================
    # 13. EXCEL REPORT GENERATION
    # ==========================================

    st.divider()

    st.header("📥 Download Excel Report")

    excel_buffer = BytesIO()

    summary_df = pd.DataFrame({
        "Project Details": [
            "File Name",
            "Selected Pages",
            "Analysis Date",
            "Total Pages Analysed",
            "Total Objects Detected",
            "Different Object Categories",
            "Preliminary Estimated Cost"
        ],
        "Value": [
            st.session_state.source_filename,
            ", ".join(
                map(str, st.session_state.selected_pages)
            ),
            datetime.now().strftime("%d-%m-%Y %H:%M"),
            len(page_results),
            total_objects,
            total_categories,
            total_cost
        ]
    })

    # Create workbook with five sheets.
    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        overall_df.to_excel(
            writer,
            sheet_name="Overall Summary",
            index=False
        )

        pagewise_df.to_excel(
            writer,
            sheet_name="Page-wise Report",
            index=False
        )

        detailed_df.to_excel(
            writer,
            sheet_name="Detailed Detection",
            index=False
        )

        cost_df.to_excel(
            writer,
            sheet_name="Cost Estimate",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="Project Summary",
            index=False
        )

    excel_buffer.seek(0)

    st.download_button(
        label="📊 Download Complete Excel Report",
        data=excel_buffer.getvalue(),
        file_name="Floor_Plan_Analysis_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "Note: Object detection and cost estimation are preliminary. "
    "All quantities and costs should be verified by a qualified engineer."
)

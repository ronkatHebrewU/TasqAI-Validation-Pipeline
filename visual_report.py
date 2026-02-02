import matplotlib.pyplot as plt
import numpy as np

def generate_report_graph(raw_total, raw_correct, filtered_total, filtered_correct, vlm_total=0, vlm_passed=0, human_total=0):
    """
    Generates a bar chart comparing Raw vs Filtered vs VLM Audited vs Human Intervention.
    """
    labels = ['Raw', 'Filtered', 'VLM Audit', 'Human Review']
    total_counts = [raw_total, filtered_total, vlm_total, human_total]
    correct_counts = [raw_correct, filtered_correct, vlm_passed, 0] # Human review doesn't have 'correct' yet

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], total_counts, width, label='Total')
    rects2 = ax.bar([i + width/2 for i in x], correct_counts, width, label='Correct/Passed')

    ax.set_ylabel('Count')
    ax.set_title('Detection Accuracy Pipeline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig('accuracy_report.png')
    print("Graph saved to accuracy_report.png")

def generate_pipeline_story_graph():
    """
    Generates a comparison chart for pipeline performance showing Precision and Recall.
    Uses mock data for demonstration.
    """
    groups = ['Raw YOLO', 'Filtered (High Conf)', 'Final (VLM Verified)']
    precision_scores = [0.1, 0.6, 0.95]
    recall_scores = [0.8, 0.4, 0.9]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precision_scores, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x + width/2, recall_scores, width, label='Recall', color='lightgreen')

    ax.set_ylabel('Score')
    ax.set_title('Pipeline Evolution: From Raw AI to Verified Insights')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.1)  # Set y-axis limit slightly above 1 for better visibility
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig('pipeline_story.png')
    print("Graph saved to pipeline_story.png")

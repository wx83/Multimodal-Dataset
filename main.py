from langgraph.graph import StateGraph, START, END

from state import AVState
from nodes import (
    av_caption_generation,
    sounding_object_extraction,
    target_object_segmentation,
    effect_erase_inpainting,
    inpainted_video_check,
    samaudio_best_of_remove,
    audio_removal_check,
    av_quality_enhancement,
    paired_av_output,
    discard_sample,
)
from routes import (
    route_after_object_extraction,
    route_after_mask_check,
    route_after_visual_check,
    route_after_audio_check,
)


def build_graph():
    """Build and compile the minimal AV data curation LangGraph."""
    builder = StateGraph(AVState)

    builder.add_node("av_caption_generation", av_caption_generation)
    builder.add_node("sounding_object_extraction", sounding_object_extraction)
    builder.add_node("target_object_segmentation", target_object_segmentation)
    builder.add_node("effect_erase_inpainting", effect_erase_inpainting)
    builder.add_node("inpainted_video_check", inpainted_video_check)
    builder.add_node("samaudio_best_of_remove", samaudio_best_of_remove)
    builder.add_node("audio_removal_check", audio_removal_check)
    builder.add_node("av_quality_enhancement", av_quality_enhancement)
    builder.add_node("paired_av_output", paired_av_output)
    builder.add_node("discard_sample", discard_sample)

    builder.add_edge(START, "av_caption_generation")
    builder.add_edge("av_caption_generation", "sounding_object_extraction")

    builder.add_conditional_edges(
        "sounding_object_extraction",
        route_after_object_extraction,
        {
            "continue": "target_object_segmentation",
            "discard": "discard_sample",
        },
    )

    builder.add_conditional_edges(
        "target_object_segmentation",
        route_after_mask_check,
        {
            "continue": "effect_erase_inpainting",
            "discard": "discard_sample",
        },
    )

    builder.add_edge("effect_erase_inpainting", "inpainted_video_check")

    builder.add_conditional_edges(
        "inpainted_video_check",
        route_after_visual_check,
        {
            "continue": "samaudio_best_of_remove",
            "discard": "discard_sample",
        },
    )

    builder.add_edge("samaudio_best_of_remove", "audio_removal_check")

    builder.add_conditional_edges(
        "audio_removal_check",
        route_after_audio_check,
        {
            "continue": "av_quality_enhancement",
            "discard": "discard_sample",
        },
    )

    builder.add_edge("av_quality_enhancement", "paired_av_output")
    builder.add_edge("paired_av_output", END)
    builder.add_edge("discard_sample", END)

    return builder.compile()


def run_one_mock_sample():
    graph = build_graph()

    initial_state = {
        "sample_id": "sample_000001",
        "av_pair_path": "data/raw/sample_000001.mp4",
        "retry_count": 0,
        "max_retries": 0,
        "status": "running",
        "mock_mask_area_ratio": 0.22,
        "mock_visual_removal_score": 0.90,
        "mock_audio_removal_score": 0.90,
    }

    print("\n========== GRAPH MERMAID ==========")
    print(graph.get_graph().draw_mermaid())

    print("\n========== STREAM UPDATES ==========")
    final_state = None
    for state in graph.stream(initial_state, stream_mode="values"):
        final_state = state
        print(state)

    print("\n========== FINAL STATE ==========")
    print(final_state)


if __name__ == "__main__":
    run_one_mock_sample()

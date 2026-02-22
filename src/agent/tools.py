"""
Pure Function Tools for CodeAct Agent.

All tools are pure functions (no side effects, no external dependencies).
These functions are available to the LLM-generated code in the sandbox.

DATA SCHEMA:
The 'data' variable contains frame-level productivity data with this structure:
{
    "total_frames": int,           # Total number of frames in the video
    "frames": [                    # List of frame dictionaries
        {
            "frame_index": int,              # 1-based frame number
            "timestamp_sec": float,          # Time in seconds from video start
            "wearer_productivity_status": str,  # "ACTIVE" or "IDLE"
            "scene_activity_status": str,    # Scene-level activity status
            "task_name": str,                # e.g., "align_or_fit_pipe", "cut_material"
            "task_family": str,              # e.g., "positioning_alignment", "cutting"
            "task_confidence": float,        # 0-1 confidence score
            "scene": str,                    # e.g., "plumbing", "electrical"
            "scene_confidence": float,       # 0-1 confidence score
            "motion": str,                   # e.g., "unknown", "stationary", "walking"
            "primary_tool": str,             # e.g., "hand pipe cutter torch", "drill"
            "active_interaction": bool,      # True if actively using tool
            "hands_count": int,              # Number of hands detected
            "tools_count": int,              # Number of tools detected
            "interactions_count": int        # Number of hand-tool interactions
        },
        ...
    ]
}
"""

from typing import Dict, Any, List, Optional


# ============================================================================
# TIME PARSING & FORMATTING TOOLS
# ============================================================================

def parse_time_string(time_str: str) -> float:
    """
    Convert a human-readable time string to seconds.

    USE THIS WHEN: You need to convert user-provided time values like "11:00"
    or "1:30:00" to seconds for use with other time-based functions.

    SUPPORTED FORMATS:
    - "HH:MM:SS" (e.g., "1:30:00" = 5400 seconds)
    - "MM:SS" (e.g., "11:00" = 660 seconds, "5:30" = 330 seconds)
    - Plain seconds (e.g., "90" = 90 seconds)

    IMPORTANT: "11:00" means 11 minutes and 0 seconds (660 seconds), NOT 11 hours!

    Args:
        time_str: Time string in one of the supported formats

    Returns:
        float: Time converted to seconds

    Example:
        start = parse_time_string("11:00")  # Returns 660.0
        end = parse_time_string("13:00")    # Returns 780.0
    """
    time_str = str(time_str).strip()
    parts = time_str.split(':')

    if len(parts) == 3:
        # HH:MM:SS
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # MM:SS
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        # Just seconds
        return float(time_str)


def format_duration(seconds: float) -> str:
    """
    Convert seconds to a human-readable duration string.

    USE THIS WHEN: You need to display time durations in a user-friendly format
    for the final answer.

    Args:
        seconds: Duration in seconds (can be float or int)

    Returns:
        str: Formatted duration like "2h 15m 30s", "5m 30s", or "45s"

    Example:
        format_duration(3661)  # Returns "1h 1m 1s"
        format_duration(330)   # Returns "5m 30s"
        format_duration(45)    # Returns "45s"
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ============================================================================
# TIME-BASED QUERY TOOLS
# ============================================================================

def get_frames_in_time_range(
    data: Dict,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[Dict]:
    """
    Get all frames within a specified time range.

    USE THIS WHEN: You need to retrieve raw frame data for a specific time period
    to perform custom analysis or iteration.

    Args:
        data: The main data dictionary containing 'frames' list
        start_time: Start time in seconds. Use None or 0 for beginning of video.
        end_time: End time in seconds. Use None for end of video.

    Returns:
        List[Dict]: List of frame dictionaries within the time range.
        Each frame contains: frame_index, timestamp_sec, wearer_productivity_status,
        task_name, task_family, scene, primary_tool, active_interaction, etc.

    Example:
        # Get frames between 11:00 and 13:00 (660-780 seconds)
        frames = get_frames_in_time_range(data, 660, 780)
        for frame in frames:
            print(frame['task_name'], frame['wearer_productivity_status'])
    """
    frames = data.get('frames', [])
    result = []

    # Handle None values - None means no limit
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = float('inf')

    for frame in frames:
        timestamp = frame.get('timestamp_sec', 0)
        if start_time <= timestamp <= end_time:
            result.append(frame)

    return result


def calculate_idle_time_in_range(
    data: Dict,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate idle and active time statistics within a time range.

    USE THIS WHEN: User asks about idle time, productivity, active time, or
    work/rest balance within a specific time period or the entire video.

    Args:
        data: The main data dictionary containing 'frames' list
        start_time: Start time in seconds. Use None for beginning of video.
        end_time: End time in seconds. Use None for end of video.

    Returns:
        Dict with these keys:
        - idle_frames (int): Number of frames where status was IDLE
        - active_frames (int): Number of frames where status was ACTIVE
        - total_frames (int): Total frames in the range
        - idle_time_seconds (float): Estimated idle time in seconds
        - active_time_seconds (float): Estimated active time in seconds
        - idle_percentage (float): Percentage of time idle (0-100)
        - active_percentage (float): Percentage of time active (0-100)
        - idle_time_formatted (str): Human-readable idle time (e.g., "5m 30s")
        - active_time_formatted (str): Human-readable active time

    Example:
        # Get idle time for entire video
        stats = calculate_idle_time_in_range(data, None, None)
        print(f"Idle: {stats['idle_time_formatted']} ({stats['idle_percentage']}%)")

        # Get idle time between 11:00-13:00
        stats = calculate_idle_time_in_range(data, 660, 780)
    """
    frames_in_range = get_frames_in_time_range(data, start_time, end_time)

    if not frames_in_range:
        return {
            'idle_frames': 0,
            'active_frames': 0,
            'total_frames': 0,
            'idle_time_seconds': 0,
            'active_time_seconds': 0,
            'total_time_seconds': 0,
            'idle_percentage': 0,
            'active_percentage': 0,
            'idle_time_formatted': '0s',
            'active_time_formatted': '0s',
            'message': 'No frames found in the specified time range'
        }

    idle_frames = 0
    active_frames = 0

    for frame in frames_in_range:
        status = frame.get('wearer_productivity_status', '').upper()
        if status == 'IDLE':
            idle_frames += 1
        else:  # ACTIVE or any other status
            active_frames += 1

    total_frames = len(frames_in_range)

    # Calculate time based on frame timestamps
    if len(frames_in_range) >= 2:
        time_span = frames_in_range[-1].get('timestamp_sec', 0) - frames_in_range[0].get('timestamp_sec', 0)
        frame_duration = time_span / (len(frames_in_range) - 1) if len(frames_in_range) > 1 else 0.033
    else:
        frame_duration = 0.033  # Default ~30fps

    idle_time_seconds = idle_frames * frame_duration
    active_time_seconds = active_frames * frame_duration
    total_time_seconds = total_frames * frame_duration

    return {
        'idle_frames': idle_frames,
        'active_frames': active_frames,
        'total_frames': total_frames,
        'idle_time_seconds': round(idle_time_seconds, 2),
        'active_time_seconds': round(active_time_seconds, 2),
        'total_time_seconds': round(total_time_seconds, 2),
        'idle_percentage': round((idle_frames / total_frames * 100) if total_frames > 0 else 0, 1),
        'active_percentage': round((active_frames / total_frames * 100) if total_frames > 0 else 0, 1),
        'idle_time_formatted': format_duration(idle_time_seconds),
        'active_time_formatted': format_duration(active_time_seconds),
        'total_time_formatted': format_duration(total_time_seconds),
    }


def get_productivity_status_in_range(
    data: Dict,
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """
    Alias for calculate_idle_time_in_range. Gets productivity breakdown in a time range.

    USE THIS WHEN: Same as calculate_idle_time_in_range - for questions about
    productivity, work time, or active/idle breakdown.

    Args:
        data: The main data dictionary
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Same as calculate_idle_time_in_range - dict with idle/active statistics.
    """
    return calculate_idle_time_in_range(data, start_time, end_time)


# ============================================================================
# TASK & ACTIVITY QUERY TOOLS
# ============================================================================

def get_tasks_in_time_range(
    data: Dict,
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """
    Get all tasks performed within a time range with detailed statistics.

    USE THIS WHEN: User asks "what tasks were performed?", "what was the worker doing?",
    or wants a breakdown of activities/tasks in a specific time period.

    Args:
        data: The main data dictionary
        start_time: Start time in seconds (use 0 for beginning)
        end_time: End time in seconds (use float('inf') for end)

    Returns:
        Dict with these keys:
        - tasks (list): List of task dicts, each containing:
            - task (str): Task name like "align_or_fit_pipe"
            - frame_count (int): Number of frames with this task
            - time_seconds (float): Time spent on task
            - time_formatted (str): Human-readable time
            - percentage (float): Percentage of time on this task
        - task_families (list): Similar breakdown by task family
        - total_tasks (int): Number of unique tasks
        - total_families (int): Number of unique task families
        - time_range (str): Formatted time range

    Example:
        # What tasks were done between 5:00-10:00?
        result = get_tasks_in_time_range(data, 300, 600)
        for task in result['tasks']:
            print(f"{task['task']}: {task['time_formatted']} ({task['percentage']}%)")
    """
    frames_in_range = get_frames_in_time_range(data, start_time, end_time)

    task_counts = {}
    task_families = {}

    for frame in frames_in_range:
        task = frame.get('task_name', 'unknown')
        family = frame.get('task_family', 'unknown')

        task_counts[task] = task_counts.get(task, 0) + 1
        task_families[family] = task_families.get(family, 0) + 1

    frame_duration = _estimate_frame_duration(frames_in_range)

    tasks_list = [
        {
            'task': task,
            'frame_count': count,
            'time_seconds': round(count * frame_duration, 2),
            'time_formatted': format_duration(count * frame_duration),
            'percentage': round(count / len(frames_in_range) * 100, 1) if frames_in_range else 0,
        }
        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    families_list = [
        {
            'family': family,
            'frame_count': count,
            'time_seconds': round(count * frame_duration, 2),
            'time_formatted': format_duration(count * frame_duration),
            'percentage': round(count / len(frames_in_range) * 100, 1) if frames_in_range else 0,
        }
        for family, count in sorted(task_families.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'tasks': tasks_list,
        'task_families': families_list,
        'unique_tasks': len(task_counts),
        'unique_families': len(task_families),
        'time_range': f"{format_duration(start_time)} - {format_duration(end_time)}",
    }


def get_tools_used_in_time_range(
    data: Dict,
    start_time: float,
    end_time: float,
) -> Dict[str, Any]:
    """
    Get all tools used within a time range with usage statistics.

    USE THIS WHEN: User asks "what tools were used?", "which equipment was used?",
    or wants to know about tool usage during a specific period.

    NOTE: Only counts frames where active_interaction=True (actively using the tool).

    Args:
        data: The main data dictionary
        start_time: Start time in seconds (use 0 for beginning)
        end_time: End time in seconds (use float('inf') for all)

    Returns:
        Dict with these keys:
        - tools (list): List of tool dicts, each containing:
            - tool (str): Tool name like "drill", "hand pipe cutter torch"
            - frame_count (int): Number of frames actively using this tool
            - time_seconds (float): Time spent using tool
            - time_formatted (str): Human-readable time
            - percentage (float): Percentage of time using this tool
        - total_unique_tools (int): Number of different tools used
        - total_frames_with_active_tools (int): Frames with active tool use
        - time_range (str): Formatted time range

    Example:
        # What tools were used in the entire video?
        result = get_tools_used_in_time_range(data, 0, float('inf'))
        for tool in result['tools']:
            print(f"{tool['tool']}: used for {tool['time_formatted']}")
    """
    frames_in_range = get_frames_in_time_range(data, start_time, end_time)

    tool_counts = {}

    for frame in frames_in_range:
        tool = frame.get('primary_tool')
        if tool and frame.get('active_interaction', False):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    frame_duration = _estimate_frame_duration(frames_in_range)

    tools_list = [
        {
            'tool': tool,
            'frame_count': count,
            'time_seconds': round(count * frame_duration, 2),
            'time_formatted': format_duration(count * frame_duration),
            'percentage': round(count / len(frames_in_range) * 100, 1) if frames_in_range else 0,
        }
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'tools': tools_list,
        'unique_tools': len(tool_counts),
        'total_frames_with_active_tools': sum(tool_counts.values()),
        'time_range': f"{format_duration(start_time)} - {format_duration(end_time)}",
    }


def get_scene_breakdown(
    data: Dict,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get breakdown of work scenes/locations with time spent in each.

    USE THIS WHEN: User asks about locations, work areas, where work happened,
    or scene distribution (e.g., "How much time in plumbing vs electrical?").

    Args:
        data: The main data dictionary
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)

    Returns:
        Dict with these keys:
        - scenes (list): List of scene dicts, each containing:
            - scene (str): Scene name like "plumbing", "electrical"
            - frame_count (int): Frames in this scene
            - time_seconds (float): Time spent in scene
            - time_formatted (str): Human-readable time
            - percentage (float): Percentage of time in this scene
        - unique_scenes (int): Number of different scenes
        - total_frames (int): Total frames analyzed

    Example:
        result = get_scene_breakdown(data)
        for scene in result['scenes']:
            print(f"{scene['scene']}: {scene['time_formatted']} ({scene['percentage']}%)")
    """
    frames = data.get('frames', [])

    if start_time is not None or end_time is not None:
        start = start_time if start_time is not None else 0
        end = end_time if end_time is not None else float('inf')
        frames = [f for f in frames if start <= f.get('timestamp_sec', 0) <= end]

    scene_counts = {}

    for frame in frames:
        scene = frame.get('scene', 'unknown')
        scene_counts[scene] = scene_counts.get(scene, 0) + 1

    frame_duration = _estimate_frame_duration(frames)

    scenes_list = [
        {
            'scene': scene,
            'frame_count': count,
            'time_seconds': round(count * frame_duration, 2),
            'time_formatted': format_duration(count * frame_duration),
            'percentage': round(count / len(frames) * 100, 1) if frames else 0,
        }
        for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'scenes': scenes_list,
        'unique_scenes': len(scene_counts),
        'total_frames': len(frames),
    }


def get_motion_breakdown(
    data: Dict,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get breakdown of motion types (stationary, walking, etc.) with time spent.

    USE THIS WHEN: User asks about movement patterns, walking time, stationary time,
    or wants to understand how the worker moved during the session.

    Args:
        data: The main data dictionary
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)

    Returns:
        Dict with these keys:
        - motions (list): List of motion dicts, each containing:
            - motion (str): Motion type like "stationary", "walking", "unknown"
            - frame_count (int): Frames with this motion
            - time_seconds (float): Time in this motion state
            - time_formatted (str): Human-readable time
            - percentage (float): Percentage of time in this motion
        - unique_motions (int): Number of different motion types
        - total_frames (int): Total frames analyzed

    Example:
        result = get_motion_breakdown(data)
        for motion in result['motions']:
            print(f"{motion['motion']}: {motion['percentage']}%")
    """
    frames = data.get('frames', [])

    if start_time is not None or end_time is not None:
        start = start_time if start_time is not None else 0
        end = end_time if end_time is not None else float('inf')
        frames = [f for f in frames if start <= f.get('timestamp_sec', 0) <= end]

    motion_counts = {}

    for frame in frames:
        motion = frame.get('motion', 'unknown')
        motion_counts[motion] = motion_counts.get(motion, 0) + 1

    frame_duration = _estimate_frame_duration(frames)

    motion_list = [
        {
            'motion': motion,
            'frame_count': count,
            'time_seconds': round(count * frame_duration, 2),
            'time_formatted': format_duration(count * frame_duration),
            'percentage': round(count / len(frames) * 100, 1) if frames else 0,
        }
        for motion, count in sorted(motion_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'motions': motion_list,
        'unique_motions': len(motion_counts),
        'total_frames': len(frames),
    }


# ============================================================================
# FRAME-LEVEL QUERY TOOLS
# ============================================================================

def get_frame_at_time(
    data: Dict,
    time_seconds: float,
) -> Dict[str, Any]:
    """
    Get the frame closest to a specific timestamp.

    USE THIS WHEN: User asks about a specific time point like "what was happening
    at 5 minutes?" or "show me the activity at 2:30".

    Args:
        data: The main data dictionary
        time_seconds: Target time in seconds (use parse_time_string to convert "MM:SS")

    Returns:
        Dict with these keys:
        - found (bool): True if a frame was found
        - frame (dict): The frame data if found, containing all frame fields:
            frame_index, timestamp_sec, wearer_productivity_status, task_name,
            task_family, scene, primary_tool, active_interaction, etc.
        - time_difference_sec (float): How far the found frame is from requested time
        - message (str): Error message if not found

    IMPORTANT: Access frame data via result['frame']['field_name']

    Example:
        # What was happening at 5:00 (300 seconds)?
        result = get_frame_at_time(data, 300)
        if result['found']:
            frame = result['frame']
            print(f"Task: {frame['task_name']}, Status: {frame['wearer_productivity_status']}")
    """
    frames = data.get('frames', [])

    if not frames:
        return {'found': False, 'message': 'No frames in data'}

    # Find closest frame
    closest_frame = None
    min_diff = float('inf')

    for frame in frames:
        diff = abs(frame.get('timestamp_sec', 0) - time_seconds)
        if diff < min_diff:
            min_diff = diff
            closest_frame = frame

    if closest_frame:
        return {
            'found': True,
            'frame': closest_frame,
            'time_difference_sec': round(min_diff, 3),
        }

    return {'found': False, 'message': 'No frame found'}


def get_frame_by_index(
    data: Dict,
    frame_index: int,
) -> Dict[str, Any]:
    """
    Get a specific frame by its frame index number.

    USE THIS WHEN: User asks about a specific frame number like "what was happening
    in frame 50?" or "show me frame 100".

    Args:
        data: The main data dictionary
        frame_index: Frame index to retrieve (1-based, starts from 1)

    Returns:
        Dict with these keys:
        - found (bool): True if frame was found
        - frame (dict): The frame data if found, containing:
            frame_index, timestamp_sec, wearer_productivity_status, task_name,
            task_family, scene, primary_tool, active_interaction, hands_count, etc.
        - message (str): Error message if not found
        - available_range (str): Valid frame range if not found

    IMPORTANT: Access frame data via result['frame']['field_name']

    Example:
        result = get_frame_by_index(data, 50)
        if result['found']:
            frame = result['frame']
            print(f"Frame 50 - Task: {frame['task_name']}")
            print(f"Status: {frame['wearer_productivity_status']}")
            print(f"Tool: {frame['primary_tool']}")
    """
    frames = data.get('frames', [])

    for frame in frames:
        if frame.get('frame_index') == frame_index:
            return {
                'found': True,
                'frame': frame,
            }

    return {
        'found': False,
        'message': f'Frame {frame_index} not found',
        'available_range': f"1 to {data.get('total_frames', 'unknown')}",
    }


def filter_frames_by_status(
    data: Dict,
    status: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get all frames where the worker had a specific productivity status.

    USE THIS WHEN: User wants to find all idle periods, all active periods,
    or needs detailed lists of frames by productivity status.

    Args:
        data: The main data dictionary
        status: "ACTIVE" or "IDLE" (case-insensitive)
        start_time: Optional start time in seconds (None = beginning)
        end_time: Optional end time in seconds (None = end)

    Returns:
        Dict with these keys:
        - frames (list): List of matching frame dictionaries
        - frame_count (int): Number of matching frames
        - status_filtered (str): The status that was filtered

    Example:
        # Find all idle frames
        result = filter_frames_by_status(data, "IDLE")
        print(f"Found {result['frame_count']} idle frames")

        # Find active frames between 5:00-10:00
        result = filter_frames_by_status(data, "ACTIVE", 300, 600)
    """
    frames = data.get('frames', [])

    result_frames = []
    for frame in frames:
        # Check time range
        timestamp = frame.get('timestamp_sec', 0)
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue

        # Check status
        frame_status = frame.get('wearer_productivity_status', '').upper()
        if frame_status == status.upper():
            result_frames.append(frame)

    return {
        'frames': result_frames,
        'frame_count': len(result_frames),
        'status_filtered': status.upper(),
    }


def filter_frames_by_task(
    data: Dict,
    task_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get all frames where a specific task was being performed.

    USE THIS WHEN: User wants to find when a specific task occurred, or needs
    detailed frame data for a particular activity.

    NOTE: Uses partial matching - "cut" will match "cut_material", "cutting", etc.

    Args:
        data: The main data dictionary
        task_name: Task name to search for (partial match, case-insensitive)
        start_time: Optional start time in seconds
        end_time: Optional end time in seconds

    Returns:
        Dict with these keys:
        - frames (list): List of matching frame dictionaries
        - frame_count (int): Number of matching frames
        - task_searched (str): The task name that was searched

    Example:
        # Find all frames where cutting was happening
        result = filter_frames_by_task(data, "cut")
        print(f"Found {result['frame_count']} frames with cutting")
    """
    frames = data.get('frames', [])

    result_frames = []
    for frame in frames:
        # Check time range
        timestamp = frame.get('timestamp_sec', 0)
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue

        # Check task (partial match)
        frame_task = frame.get('task_name', '').lower()
        if task_name.lower() in frame_task:
            result_frames.append(frame)

    return {
        'frames': result_frames,
        'frame_count': len(result_frames),
        'task_searched': task_name,
    }


def filter_frames_by_tool(
    data: Dict,
    tool_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get all frames where a specific tool was the primary tool.

    USE THIS WHEN: User wants to find when a specific tool was used, or needs
    frame data for a particular equipment usage.

    NOTE: Uses partial matching - "drill" will match "power drill", "drill press", etc.

    Args:
        data: The main data dictionary
        tool_name: Tool name to search for (partial match, case-insensitive)
        start_time: Optional start time in seconds
        end_time: Optional end time in seconds

    Returns:
        Dict with these keys:
        - frames (list): List of matching frame dictionaries
        - frame_count (int): Number of matching frames
        - tool_searched (str): The tool name that was searched

    Example:
        # Find all frames with drill usage
        result = filter_frames_by_tool(data, "drill")
        print(f"Drill appeared in {result['frame_count']} frames")
    """
    frames = data.get('frames', [])

    result_frames = []
    for frame in frames:
        # Check time range
        timestamp = frame.get('timestamp_sec', 0)
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue

        # Check tool (partial match)
        frame_tool = frame.get('primary_tool', '')
        if frame_tool and tool_name.lower() in frame_tool.lower():
            result_frames.append(frame)

    return {
        'frames': result_frames,
        'frame_count': len(result_frames),
        'tool_searched': tool_name,
    }


# ============================================================================
# SUMMARY & AGGREGATE TOOLS
# ============================================================================

def get_overall_summary(data: Dict) -> Dict[str, Any]:
    """
    Get a comprehensive summary of the entire video/session.

    USE THIS WHEN: User asks for an overview, summary, general statistics,
    or wants to understand the overall session at a high level.

    This is often the BEST STARTING POINT for general questions.

    Args:
        data: The main data dictionary

    Returns:
        Dict with these keys:
        - total_frames (int): Total number of frames
        - duration_seconds (float): Total video duration in seconds
        - duration_formatted (str): Human-readable duration (e.g., "5m 30s")
        - active_frames (int): Frames where worker was ACTIVE
        - idle_frames (int): Frames where worker was IDLE
        - active_percentage (float): Percentage of time active (0-100)
        - idle_percentage (float): Percentage of time idle (0-100)
        - unique_tasks (list): List of all unique task names
        - unique_tools (list): List of all unique tools used
        - unique_scenes (list): List of all unique scenes/locations
        - task_count (int): Number of different tasks
        - tool_count (int): Number of different tools
        - scene_count (int): Number of different scenes
        - frames_with_interaction (int): Frames with active tool interaction
        - interaction_percentage (float): Percentage of time with active interaction

    Example:
        summary = get_overall_summary(data)
        print(f"Duration: {summary['duration_formatted']}")
        print(f"Active: {summary['active_percentage']}%")
        print(f"Tasks: {summary['unique_tasks']}")
        print(f"Tools: {summary['unique_tools']}")
    """
    frames = data.get('frames', [])
    total_frames = data.get('total_frames', len(frames))

    if not frames:
        return {'message': 'No frames in data', 'total_frames': 0}

    # Calculate duration
    if len(frames) >= 2:
        duration = frames[-1].get('timestamp_sec', 0) - frames[0].get('timestamp_sec', 0)
    else:
        duration = 0

    # Count statuses
    active_count = sum(1 for f in frames if f.get('wearer_productivity_status', '').upper() == 'ACTIVE')
    idle_count = total_frames - active_count

    # Get unique values
    unique_tasks = set(f.get('task_name', 'unknown') for f in frames)
    unique_tools = set(f.get('primary_tool', '') for f in frames if f.get('primary_tool'))
    unique_scenes = set(f.get('scene', 'unknown') for f in frames)

    # Count interactions
    interaction_frames = sum(1 for f in frames if f.get('active_interaction', False))

    return {
        'total_frames': total_frames,
        'duration_seconds': round(duration, 2),
        'duration_formatted': format_duration(duration),
        'active_frames': active_count,
        'idle_frames': idle_count,
        'active_percentage': round(active_count / total_frames * 100, 1) if total_frames > 0 else 0,
        'idle_percentage': round(idle_count / total_frames * 100, 1) if total_frames > 0 else 0,
        'unique_tasks': list(unique_tasks),
        'unique_tools': list(unique_tools),
        'unique_scenes': list(unique_scenes),
        'task_count': len(unique_tasks),
        'tool_count': len(unique_tools),
        'scene_count': len(unique_scenes),
        'frames_with_interaction': interaction_frames,
        'interaction_percentage': round(interaction_frames / total_frames * 100, 1) if total_frames > 0 else 0,
    }


def get_all_unique_tasks(data: Dict) -> Dict[str, Any]:
    """
    Get a list of all unique tasks with their occurrence counts.

    USE THIS WHEN: User asks "what tasks were performed?", "list all activities",
    or needs a simple task inventory without time range filtering.

    Args:
        data: The main data dictionary

    Returns:
        Dict with these keys:
        - tasks (list): List of task dicts sorted by frequency, each containing:
            - task (str): Task name
            - count (int): Number of frames with this task
        - total_unique (int): Total number of unique tasks

    Example:
        result = get_all_unique_tasks(data)
        print(f"Found {result['total_unique']} unique tasks:")
        for task in result['tasks']:
            print(f"  - {task['task']}: {task['count']} frames")
    """
    frames = data.get('frames', [])
    task_counts = {}

    for frame in frames:
        task = frame.get('task_name', 'unknown')
        task_counts[task] = task_counts.get(task, 0) + 1

    tasks_list = [
        {'task': task, 'count': count}
        for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'tasks': tasks_list,
        'total_unique': len(tasks_list),
    }


def get_all_unique_tools(data: Dict) -> Dict[str, Any]:
    """
    Get a list of all unique tools with their occurrence counts.

    USE THIS WHEN: User asks "what tools were used?", "list all equipment",
    or needs a simple tool inventory.

    Args:
        data: The main data dictionary

    Returns:
        Dict with these keys:
        - tools (list): List of tool dicts sorted by frequency, each containing:
            - tool (str): Tool name
            - count (int): Number of frames with this tool
        - total_unique (int): Total number of unique tools

    Example:
        result = get_all_unique_tools(data)
        print(f"Found {result['total_unique']} unique tools:")
        for tool in result['tools']:
            print(f"  - {tool['tool']}: {tool['count']} frames")
    """
    frames = data.get('frames', [])
    tool_counts = {}

    for frame in frames:
        tool = frame.get('primary_tool')
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    tools_list = [
        {'tool': tool, 'count': count}
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        'tools': tools_list,
        'total_unique': len(tools_list),
    }


# ============================================================================
# ADVANCED ANALYTICS TOOLS (Ported from session-level agent)
# ============================================================================

def find_idle_periods(
    data: Dict,
    min_duration_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Find continuous periods where the worker was idle/unproductive.

    USE THIS WHEN: User asks about idle periods, breaks, downtime, unproductive
    stretches, or wants to identify when work stopped.

    This tool identifies CONTINUOUS idle stretches, not just individual frames.

    Args:
        data: The main data dictionary
        min_duration_seconds: Minimum duration to report as an idle period (default: 2s)

    Returns:
        Dict with these keys:
        - idle_periods (list): List of idle period dicts, each containing:
            - start_time (float): Start time in seconds
            - start_time_formatted (str): Human-readable start time
            - end_time (float): End time in seconds
            - end_time_formatted (str): Human-readable end time
            - duration_seconds (float): Duration of idle period
            - duration_formatted (str): Human-readable duration
            - frame_count (int): Number of frames in this idle period
        - total_idle_periods (int): Number of idle periods found
        - total_idle_time_seconds (float): Total time spent idle
        - total_idle_time_formatted (str): Human-readable total idle time
        - longest_idle_period (dict): The longest idle period found
        - average_idle_duration (float): Average duration of idle periods

    Example:
        result = find_idle_periods(data, min_duration_seconds=5)
        print(f"Found {result['total_idle_periods']} idle periods")
        print(f"Total idle time: {result['total_idle_time_formatted']}")
        for period in result['idle_periods']:
            print(f"  {period['start_time_formatted']} - {period['duration_formatted']}")
    """
    frames = data.get('frames', [])
    if not frames:
        return {'message': 'No frames in data', 'idle_periods': [], 'total_idle_periods': 0}

    # Sort frames by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.get('timestamp_sec', 0))

    idle_periods = []
    current_idle_start = None
    current_idle_frames = []

    for frame in sorted_frames:
        status = frame.get('wearer_productivity_status', '').upper()
        timestamp = frame.get('timestamp_sec', 0)

        if status == 'IDLE':
            if current_idle_start is None:
                current_idle_start = timestamp
                current_idle_frames = [frame]
            else:
                current_idle_frames.append(frame)
        else:
            # End of idle period
            if current_idle_start is not None and current_idle_frames:
                idle_end = current_idle_frames[-1].get('timestamp_sec', 0)
                duration = idle_end - current_idle_start

                if duration >= min_duration_seconds:
                    idle_periods.append({
                        'start_time': current_idle_start,
                        'start_time_formatted': format_duration(current_idle_start),
                        'end_time': idle_end,
                        'end_time_formatted': format_duration(idle_end),
                        'duration_seconds': round(duration, 2),
                        'duration_formatted': format_duration(duration),
                        'frame_count': len(current_idle_frames),
                    })

            current_idle_start = None
            current_idle_frames = []

    # Check if session ended during idle
    if current_idle_start is not None and current_idle_frames:
        idle_end = current_idle_frames[-1].get('timestamp_sec', 0)
        duration = idle_end - current_idle_start
        if duration >= min_duration_seconds:
            idle_periods.append({
                'start_time': current_idle_start,
                'start_time_formatted': format_duration(current_idle_start),
                'end_time': idle_end,
                'end_time_formatted': format_duration(idle_end),
                'duration_seconds': round(duration, 2),
                'duration_formatted': format_duration(duration),
                'frame_count': len(current_idle_frames),
            })

    total_idle_time = sum(p['duration_seconds'] for p in idle_periods)
    longest_period = max(idle_periods, key=lambda p: p['duration_seconds']) if idle_periods else None

    return {
        'idle_periods': idle_periods,
        'total_idle_periods': len(idle_periods),
        'total_idle_time_seconds': round(total_idle_time, 2),
        'total_idle_time_formatted': format_duration(total_idle_time),
        'longest_idle_period': longest_period,
        'average_idle_duration': round(total_idle_time / len(idle_periods), 2) if idle_periods else 0,
        'average_idle_duration_formatted': format_duration(total_idle_time / len(idle_periods)) if idle_periods else '0s',
    }


def compare_time_periods(
    data: Dict,
    period1_start: float,
    period1_end: float,
    period2_start: float,
    period2_end: float,
) -> Dict[str, Any]:
    """
    Compare productivity between two time periods.

    USE THIS WHEN: User asks to compare different parts of the session, like
    "compare first half vs second half", "was morning more productive than afternoon?",
    or "compare productivity before and after lunch".

    Args:
        data: The main data dictionary
        period1_start: Start time of first period in seconds
        period1_end: End time of first period in seconds
        period2_start: Start time of second period in seconds
        period2_end: End time of second period in seconds

    Returns:
        Dict with these keys:
        - period1 (dict): Stats for first period
            - time_range (str): Formatted time range
            - active_percentage (float): Percentage of time active
            - idle_percentage (float): Percentage of time idle
            - unique_tasks (list): Tasks performed
            - unique_tools (list): Tools used
        - period2 (dict): Stats for second period (same structure)
        - comparison (dict):
            - more_productive (str): "period1" or "period2"
            - productivity_difference (float): Difference in active percentage
            - analysis (str): Brief analysis of the comparison

    Example:
        # Compare first 5 minutes vs second 5 minutes
        result = compare_time_periods(data, 0, 300, 300, 600)
        print(f"Period 1: {result['period1']['active_percentage']}% active")
        print(f"Period 2: {result['period2']['active_percentage']}% active")
        print(f"More productive: {result['comparison']['more_productive']}")
    """
    # Get stats for each period
    stats1 = calculate_idle_time_in_range(data, period1_start, period1_end)
    stats2 = calculate_idle_time_in_range(data, period2_start, period2_end)

    frames1 = get_frames_in_time_range(data, period1_start, period1_end)
    frames2 = get_frames_in_time_range(data, period2_start, period2_end)

    # Get unique tasks and tools for each period
    tasks1 = set(f.get('task_name', 'unknown') for f in frames1)
    tasks2 = set(f.get('task_name', 'unknown') for f in frames2)
    tools1 = set(f.get('primary_tool', '') for f in frames1 if f.get('primary_tool'))
    tools2 = set(f.get('primary_tool', '') for f in frames2 if f.get('primary_tool'))

    period1_data = {
        'time_range': f"{format_duration(period1_start)} - {format_duration(period1_end)}",
        'duration_seconds': period1_end - period1_start,
        'duration_formatted': format_duration(period1_end - period1_start),
        'active_percentage': stats1['active_percentage'],
        'idle_percentage': stats1['idle_percentage'],
        'active_time_formatted': stats1['active_time_formatted'],
        'idle_time_formatted': stats1['idle_time_formatted'],
        'total_frames': stats1['total_frames'],
        'unique_tasks': list(tasks1),
        'unique_tools': list(tools1),
    }

    period2_data = {
        'time_range': f"{format_duration(period2_start)} - {format_duration(period2_end)}",
        'duration_seconds': period2_end - period2_start,
        'duration_formatted': format_duration(period2_end - period2_start),
        'active_percentage': stats2['active_percentage'],
        'idle_percentage': stats2['idle_percentage'],
        'active_time_formatted': stats2['active_time_formatted'],
        'idle_time_formatted': stats2['idle_time_formatted'],
        'total_frames': stats2['total_frames'],
        'unique_tasks': list(tasks2),
        'unique_tools': list(tools2),
    }

    # Determine which period was more productive
    diff = stats2['active_percentage'] - stats1['active_percentage']
    more_productive = "period2" if diff > 0 else "period1" if diff < 0 else "equal"

    if abs(diff) < 5:
        analysis = "Both periods had similar productivity levels."
    elif diff > 0:
        analysis = f"Period 2 was more productive by {abs(diff):.1f} percentage points."
    else:
        analysis = f"Period 1 was more productive by {abs(diff):.1f} percentage points."

    return {
        'period1': period1_data,
        'period2': period2_data,
        'comparison': {
            'more_productive': more_productive,
            'productivity_difference': round(diff, 1),
            'analysis': analysis,
        }
    }


def detect_fatigue(data: Dict) -> Dict[str, Any]:
    """
    Detect behavioral fatigue by comparing first half vs second half of session.

    USE THIS WHEN: User asks about fatigue, tiredness, energy levels, performance
    decline, or whether the worker got tired during the session.

    Analyzes:
    - Productivity drop between halves
    - Increase in idle time
    - Changes in task patterns

    Args:
        data: The main data dictionary

    Returns:
        Dict with these keys:
        - fatigue_detected (bool): True if signs of fatigue were detected
        - fatigue_level (str): "none", "mild", "moderate", or "severe"
        - first_half (dict): Stats for first half of session
        - second_half (dict): Stats for second half of session
        - indicators (list): List of fatigue indicators found
        - productivity_drop (float): Percentage point drop in productivity
        - idle_increase (float): Percentage point increase in idle time
        - recommendation (str): Suggestion based on analysis

    Example:
        result = detect_fatigue(data)
        if result['fatigue_detected']:
            print(f"Fatigue level: {result['fatigue_level']}")
            print(f"Productivity dropped by {result['productivity_drop']}%")
    """
    frames = data.get('frames', [])
    if len(frames) < 10:
        return {
            'fatigue_detected': False,
            'fatigue_level': 'unknown',
            'message': 'Not enough data to detect fatigue (need at least 10 frames)',
        }

    # Sort frames and find midpoint
    sorted_frames = sorted(frames, key=lambda f: f.get('timestamp_sec', 0))
    min_time = sorted_frames[0].get('timestamp_sec', 0)
    max_time = sorted_frames[-1].get('timestamp_sec', 0)
    mid_time = (min_time + max_time) / 2

    # Get stats for each half
    first_half_stats = calculate_idle_time_in_range(data, min_time, mid_time)
    second_half_stats = calculate_idle_time_in_range(data, mid_time, max_time)

    # Calculate changes
    productivity_drop = first_half_stats['active_percentage'] - second_half_stats['active_percentage']
    idle_increase = second_half_stats['idle_percentage'] - first_half_stats['idle_percentage']

    # Detect fatigue indicators
    indicators = []
    fatigue_score = 0

    if productivity_drop > 10:
        indicators.append(f"Productivity dropped by {productivity_drop:.1f}% in second half")
        fatigue_score += 2
    elif productivity_drop > 5:
        indicators.append(f"Slight productivity decrease ({productivity_drop:.1f}%) in second half")
        fatigue_score += 1

    if idle_increase > 10:
        indicators.append(f"Idle time increased by {idle_increase:.1f}% in second half")
        fatigue_score += 2
    elif idle_increase > 5:
        indicators.append(f"Slight idle time increase ({idle_increase:.1f}%) in second half")
        fatigue_score += 1

    # Determine fatigue level
    if fatigue_score >= 4:
        fatigue_level = "severe"
        recommendation = "Consider taking a break or ending the work session. Fatigue significantly impacts productivity and safety."
    elif fatigue_score >= 2:
        fatigue_level = "moderate"
        recommendation = "A short break may help restore productivity levels."
    elif fatigue_score >= 1:
        fatigue_level = "mild"
        recommendation = "Minor fatigue indicators present. Monitor for further decline."
    else:
        fatigue_level = "none"
        recommendation = "No significant fatigue detected. Worker maintained consistent performance."

    return {
        'fatigue_detected': fatigue_score >= 2,
        'fatigue_level': fatigue_level,
        'first_half': {
            'time_range': f"{format_duration(min_time)} - {format_duration(mid_time)}",
            'active_percentage': first_half_stats['active_percentage'],
            'idle_percentage': first_half_stats['idle_percentage'],
        },
        'second_half': {
            'time_range': f"{format_duration(mid_time)} - {format_duration(max_time)}",
            'active_percentage': second_half_stats['active_percentage'],
            'idle_percentage': second_half_stats['idle_percentage'],
        },
        'indicators': indicators,
        'productivity_drop': round(productivity_drop, 1),
        'idle_increase': round(idle_increase, 1),
        'recommendation': recommendation,
    }


def get_productivity_trend(
    data: Dict,
    num_segments: int = 5,
) -> Dict[str, Any]:
    """
    Analyze productivity trend over the session by dividing into segments.

    USE THIS WHEN: User asks about trends, patterns over time, whether productivity
    improved or declined, or wants to understand the trajectory of work.

    Args:
        data: The main data dictionary
        num_segments: Number of time segments to analyze (default: 5)

    Returns:
        Dict with these keys:
        - trend (str): "improving", "declining", or "stable"
        - trend_strength (str): "strong", "moderate", or "slight"
        - segments (list): List of segment stats with active_percentage
        - slope (float): Rate of change (positive = improving)
        - start_productivity (float): Productivity at start
        - end_productivity (float): Productivity at end
        - analysis (str): Description of the trend

    Example:
        result = get_productivity_trend(data)
        print(f"Trend: {result['trend']} ({result['trend_strength']})")
        print(f"Analysis: {result['analysis']}")
    """
    frames = data.get('frames', [])
    if len(frames) < num_segments * 2:
        return {
            'trend': 'unknown',
            'message': f'Not enough data for trend analysis (need at least {num_segments * 2} frames)',
        }

    # Sort frames and divide into segments
    sorted_frames = sorted(frames, key=lambda f: f.get('timestamp_sec', 0))
    min_time = sorted_frames[0].get('timestamp_sec', 0)
    max_time = sorted_frames[-1].get('timestamp_sec', 0)
    segment_duration = (max_time - min_time) / num_segments

    segments = []
    for i in range(num_segments):
        start = min_time + i * segment_duration
        end = min_time + (i + 1) * segment_duration
        stats = calculate_idle_time_in_range(data, start, end)
        segments.append({
            'segment': i + 1,
            'time_range': f"{format_duration(start)} - {format_duration(end)}",
            'active_percentage': stats['active_percentage'],
            'idle_percentage': stats['idle_percentage'],
        })

    # Calculate trend using simple linear regression
    productivities = [s['active_percentage'] for s in segments]
    n = len(productivities)
    x_mean = (n - 1) / 2
    y_mean = sum(productivities) / n

    numerator = sum((i - x_mean) * (productivities[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0

    # Determine trend direction and strength
    if slope > 2:
        trend = "improving"
        strength = "strong" if slope > 5 else "moderate"
    elif slope < -2:
        trend = "declining"
        strength = "strong" if slope < -5 else "moderate"
    else:
        trend = "stable"
        strength = "slight" if abs(slope) > 0.5 else "very"

    # Generate analysis
    if trend == "improving":
        analysis = f"Productivity improved over the session, increasing by approximately {abs(slope * (n-1)):.1f} percentage points."
    elif trend == "declining":
        analysis = f"Productivity declined over the session, decreasing by approximately {abs(slope * (n-1)):.1f} percentage points."
    else:
        analysis = "Productivity remained relatively stable throughout the session."

    return {
        'trend': trend,
        'trend_strength': strength,
        'segments': segments,
        'slope': round(slope, 2),
        'start_productivity': productivities[0],
        'end_productivity': productivities[-1],
        'change': round(productivities[-1] - productivities[0], 1),
        'analysis': analysis,
    }


def get_peak_productivity_period(
    data: Dict,
    window_seconds: float = 60.0,
) -> Dict[str, Any]:
    """
    Find the peak productivity period during the session.

    USE THIS WHEN: User asks about best performance, most productive time,
    when the worker was most efficient, or peak performance periods.

    Args:
        data: The main data dictionary
        window_seconds: Size of the sliding window in seconds (default: 60s)

    Returns:
        Dict with these keys:
        - found (bool): True if a peak period was found
        - peak_start (float): Start time of peak period in seconds
        - peak_start_formatted (str): Human-readable start time
        - peak_end (float): End time of peak period in seconds
        - peak_end_formatted (str): Human-readable end time
        - peak_productivity (float): Productivity percentage during peak
        - peak_task (str): Most common task during peak
        - peak_tool (str): Most used tool during peak
        - comparison_to_average (float): How much better than average

    Example:
        result = get_peak_productivity_period(data, window_seconds=120)
        if result['found']:
            print(f"Peak period: {result['peak_start_formatted']} - {result['peak_end_formatted']}")
            print(f"Productivity: {result['peak_productivity']}%")
    """
    frames = data.get('frames', [])
    if len(frames) < 5:
        return {'found': False, 'message': 'Not enough data to find peak period'}

    sorted_frames = sorted(frames, key=lambda f: f.get('timestamp_sec', 0))
    min_time = sorted_frames[0].get('timestamp_sec', 0)
    max_time = sorted_frames[-1].get('timestamp_sec', 0)

    # Slide window and find peak
    best_productivity = -1
    best_start = min_time
    step = window_seconds / 4  # Overlap windows

    current_start = min_time
    while current_start + window_seconds <= max_time:
        stats = calculate_idle_time_in_range(data, current_start, current_start + window_seconds)
        if stats['active_percentage'] > best_productivity:
            best_productivity = stats['active_percentage']
            best_start = current_start
        current_start += step

    # Get details for peak period
    peak_frames = get_frames_in_time_range(data, best_start, best_start + window_seconds)

    # Find most common task and tool during peak
    task_counts = {}
    tool_counts = {}
    for frame in peak_frames:
        task = frame.get('task_name', 'unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
        tool = frame.get('primary_tool')
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    peak_task = max(task_counts, key=task_counts.get) if task_counts else 'unknown'
    peak_tool = max(tool_counts, key=tool_counts.get) if tool_counts else 'none'

    # Calculate average productivity
    overall_stats = calculate_idle_time_in_range(data, None, None)
    avg_productivity = overall_stats['active_percentage']

    return {
        'found': True,
        'peak_start': round(best_start, 2),
        'peak_start_formatted': format_duration(best_start),
        'peak_end': round(best_start + window_seconds, 2),
        'peak_end_formatted': format_duration(best_start + window_seconds),
        'peak_duration_formatted': format_duration(window_seconds),
        'peak_productivity': round(best_productivity, 1),
        'peak_task': peak_task,
        'peak_tool': peak_tool,
        'average_productivity': round(avg_productivity, 1),
        'comparison_to_average': round(best_productivity - avg_productivity, 1),
    }


def get_activity_segments(
    data: Dict,
    min_segment_duration: float = 2.0,
) -> Dict[str, Any]:
    """
    Get continuous activity segments (periods of consistent activity).

    USE THIS WHEN: User asks about work segments, activity patterns, or wants
    to see how work was organized into chunks.

    Args:
        data: The main data dictionary
        min_segment_duration: Minimum duration for a segment in seconds

    Returns:
        Dict with these keys:
        - segments (list): List of activity segments, each containing:
            - start_time, end_time, duration
            - status: "ACTIVE" or "IDLE"
            - primary_task: Most common task in segment
            - primary_tool: Most common tool in segment
        - total_segments (int): Number of segments
        - average_segment_duration (float): Average segment length
        - longest_segment (dict): The longest segment

    Example:
        result = get_activity_segments(data)
        for seg in result['segments']:
            print(f"{seg['status']}: {seg['duration_formatted']} - {seg['primary_task']}")
    """
    frames = data.get('frames', [])
    if not frames:
        return {'segments': [], 'total_segments': 0, 'message': 'No frames in data'}

    sorted_frames = sorted(frames, key=lambda f: f.get('timestamp_sec', 0))

    segments = []
    current_status = None
    current_start = None
    current_frames = []

    for frame in sorted_frames:
        status = frame.get('wearer_productivity_status', '').upper()
        timestamp = frame.get('timestamp_sec', 0)

        if status != current_status:
            # Save previous segment
            if current_start is not None and current_frames:
                end_time = current_frames[-1].get('timestamp_sec', 0)
                duration = end_time - current_start

                if duration >= min_segment_duration:
                    # Find most common task and tool
                    task_counts = {}
                    tool_counts = {}
                    for f in current_frames:
                        task = f.get('task_name', 'unknown')
                        task_counts[task] = task_counts.get(task, 0) + 1
                        tool = f.get('primary_tool')
                        if tool:
                            tool_counts[tool] = tool_counts.get(tool, 0) + 1

                    segments.append({
                        'start_time': current_start,
                        'start_time_formatted': format_duration(current_start),
                        'end_time': end_time,
                        'end_time_formatted': format_duration(end_time),
                        'duration_seconds': round(duration, 2),
                        'duration_formatted': format_duration(duration),
                        'status': current_status,
                        'frame_count': len(current_frames),
                        'primary_task': max(task_counts, key=task_counts.get) if task_counts else 'unknown',
                        'primary_tool': max(tool_counts, key=tool_counts.get) if tool_counts else 'none',
                    })

            # Start new segment
            current_status = status
            current_start = timestamp
            current_frames = [frame]
        else:
            current_frames.append(frame)

    # Handle last segment
    if current_start is not None and current_frames:
        end_time = current_frames[-1].get('timestamp_sec', 0)
        duration = end_time - current_start
        if duration >= min_segment_duration:
            task_counts = {}
            tool_counts = {}
            for f in current_frames:
                task = f.get('task_name', 'unknown')
                task_counts[task] = task_counts.get(task, 0) + 1
                tool = f.get('primary_tool')
                if tool:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

            segments.append({
                'start_time': current_start,
                'start_time_formatted': format_duration(current_start),
                'end_time': end_time,
                'end_time_formatted': format_duration(end_time),
                'duration_seconds': round(duration, 2),
                'duration_formatted': format_duration(duration),
                'status': current_status,
                'frame_count': len(current_frames),
                'primary_task': max(task_counts, key=task_counts.get) if task_counts else 'unknown',
                'primary_tool': max(tool_counts, key=tool_counts.get) if tool_counts else 'none',
            })

    total_duration = sum(s['duration_seconds'] for s in segments)
    longest = max(segments, key=lambda s: s['duration_seconds']) if segments else None

    return {
        'segments': segments,
        'total_segments': len(segments),
        'total_duration_seconds': round(total_duration, 2),
        'total_duration_formatted': format_duration(total_duration),
        'average_segment_duration': round(total_duration / len(segments), 2) if segments else 0,
        'average_segment_duration_formatted': format_duration(total_duration / len(segments)) if segments else '0s',
        'longest_segment': longest,
    }


# ============================================================================
# HELPER FUNCTIONS (Internal use only, not exposed to agent)
# ============================================================================

def _estimate_frame_duration(frames: List[Dict]) -> float:
    """Estimate the duration of each frame based on timestamps."""
    if len(frames) >= 2:
        time_span = frames[-1].get('timestamp_sec', 0) - frames[0].get('timestamp_sec', 0)
        return time_span / (len(frames) - 1) if len(frames) > 1 else 0.033
    return 0.033  # Default ~30fps


# ============================================================================
# TOOL REGISTRY & DESCRIPTIONS
# ============================================================================

def get_tool_registry() -> Dict[str, callable]:
    """
    Get registry of all available tools for the CodeAct agent.

    Returns:
        Dictionary mapping tool names to functions
    """
    return {
        # Time parsing/formatting
        'parse_time_string': parse_time_string,
        'format_duration': format_duration,

        # Time-based queries
        'get_frames_in_time_range': get_frames_in_time_range,
        'calculate_idle_time_in_range': calculate_idle_time_in_range,
        'get_productivity_status_in_range': get_productivity_status_in_range,

        # Task & activity queries
        'get_tasks_in_time_range': get_tasks_in_time_range,
        'get_tools_used_in_time_range': get_tools_used_in_time_range,
        'get_scene_breakdown': get_scene_breakdown,
        'get_motion_breakdown': get_motion_breakdown,

        # Frame-level queries
        'get_frame_at_time': get_frame_at_time,
        'get_frame_by_index': get_frame_by_index,
        'filter_frames_by_status': filter_frames_by_status,
        'filter_frames_by_task': filter_frames_by_task,
        'filter_frames_by_tool': filter_frames_by_tool,

        # Summary & aggregates
        'get_overall_summary': get_overall_summary,
        'get_all_unique_tasks': get_all_unique_tasks,
        'get_all_unique_tools': get_all_unique_tools,

        # Advanced analytics
        'find_idle_periods': find_idle_periods,
        'compare_time_periods': compare_time_periods,
        'detect_fatigue': detect_fatigue,
        'get_productivity_trend': get_productivity_trend,
        'get_peak_productivity_period': get_peak_productivity_period,
        'get_activity_segments': get_activity_segments,
    }


def get_tool_docstring(tool_name: str) -> str:
    """
    Get the docstring for a specific tool by name.

    Args:
        tool_name: Name of the tool function

    Returns:
        The tool's docstring, or empty string if not found
    """
    registry = get_tool_registry()
    if tool_name in registry:
        func = registry[tool_name]
        return func.__doc__ or ""
    return ""


def get_all_tool_names() -> List[str]:
    """
    Get list of all available tool names.

    Returns:
        List of tool function names
    """
    return list(get_tool_registry().keys())


def get_tool_names_with_brief_descriptions() -> str:
    """
    Get a brief listing of all tools for the query analyzer.

    This provides a compact overview so the analyzer can select relevant tools.
    The code generator will then receive full docstrings for only the selected tools.

    Returns:
        Formatted string with tool names and one-line descriptions
    """
    registry = get_tool_registry()
    lines = ["=== AVAILABLE TOOLS ===\n"]

    for name, func in registry.items():
        # Extract first line of docstring as brief description
        doc = func.__doc__ or "No description"
        first_line = doc.strip().split('\n')[0].strip()
        lines.append(f"- {name}: {first_line}")

    lines.append("\nReturn the tool names that would help answer the query.")
    return "\n".join(lines)


def get_selected_tool_descriptions(tool_names: List[str]) -> str:
    """
    Get full docstrings for a list of specific tools.

    This is used by the code generator to get rich descriptions
    for only the tools identified as relevant by the query analyzer.

    Args:
        tool_names: List of tool names to get descriptions for

    Returns:
        Formatted string with full docstrings for selected tools
    """
    registry = get_tool_registry()
    lines = [
        "=== TOOLS FOR THIS QUERY ===",
        "",
        "The 'data' variable is pre-loaded with frame-level productivity data.",
        "Use ONLY these tools to answer the query:",
        "",
    ]

    for name in tool_names:
        if name in registry:
            func = registry[name]
            docstring = func.__doc__ or "No documentation available"
            lines.append(f"{'='*60}")
            lines.append(f"TOOL: {name}")
            lines.append(f"{'='*60}")
            lines.append(docstring.strip())
            lines.append("")

    lines.append("="*60)
    lines.append("IMPORTANT REMINDERS:")
    lines.append("- Always assign the final result to 'answer'")
    lines.append("- The 'data' variable is already available")
    lines.append("- Time strings like '11:00' mean 11 minutes, use parse_time_string()")
    lines.append("="*60)

    return "\n".join(lines)


def parse_relevant_tools(relevant_tools_str: str) -> List[str]:
    """
    Parse the relevant tools string from query analyzer into a list.

    Handles various formats:
    - Comma-separated: "tool1, tool2, tool3"
    - Newline-separated
    - With or without quotes

    Args:
        relevant_tools_str: String containing tool names

    Returns:
        List of valid tool names
    """
    registry = get_tool_registry()
    valid_tools = set(registry.keys())

    # Clean and split the string
    cleaned = relevant_tools_str.replace('"', '').replace("'", '')

    # Try comma-separated first
    if ',' in cleaned:
        candidates = [t.strip() for t in cleaned.split(',')]
    else:
        # Try newline or space-separated
        candidates = cleaned.split()

    # Filter to only valid tool names
    result = [t for t in candidates if t in valid_tools]

    # If no valid tools found, return a default set
    if not result:
        result = ['get_overall_summary']  # Safe default

    return result


def get_tool_descriptions() -> str:
    """
    Get detailed descriptions of all available tools for the LLM.

    Returns:
        Formatted string with comprehensive tool documentation
    """
    return """
=== AVAILABLE TOOLS FOR PRODUCTIVITY DATA ANALYSIS ===

The 'data' variable is pre-loaded with frame-level productivity data.
Use these tools to query and analyze the data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIME PARSING & FORMATTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

parse_time_string(time_str) -> float
  Convert time string to seconds.
  Input: "HH:MM:SS", "MM:SS", or seconds as string
  IMPORTANT: "11:00" = 11 minutes = 660 seconds (NOT 11 hours)
  Example: parse_time_string("11:00") returns 660.0

format_duration(seconds) -> str
  Convert seconds to readable format.
  Example: format_duration(3661) returns "1h 1m 1s"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDLE/ACTIVE TIME ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

calculate_idle_time_in_range(data, start_time=None, end_time=None) -> dict
  Get idle/active time statistics. Use None for entire video.
  Returns: {
    'idle_frames': int,
    'active_frames': int,
    'idle_time_seconds': float,
    'active_time_seconds': float,
    'idle_percentage': float,
    'active_percentage': float,
    'idle_time_formatted': str,  # e.g., "5m 30s"
    'active_time_formatted': str
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK & TOOL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

get_tasks_in_time_range(data, start_time, end_time) -> dict
  Get tasks performed with time breakdown.
  Returns: {
    'tasks': [{'task': str, 'time_formatted': str, 'percentage': float}, ...],
    'unique_tasks': int
  }

get_tools_used_in_time_range(data, start_time, end_time) -> dict
  Get tools used with time breakdown (only counts active interactions).
  Returns: {
    'tools': [{'tool': str, 'time_formatted': str, 'percentage': float}, ...],
    'unique_tools': int
  }

get_all_unique_tasks(data) -> dict
  Simple list of all tasks with counts.
  Returns: {'tasks': [{'task': str, 'count': int}, ...], 'total_unique': int}

get_all_unique_tools(data) -> dict
  Simple list of all tools with counts.
  Returns: {'tools': [{'tool': str, 'count': int}, ...], 'total_unique': int}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENE & MOTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

get_scene_breakdown(data, start_time=None, end_time=None) -> dict
  Get time spent in each scene/location.
  Returns: {'scenes': [{'scene': str, 'time_formatted': str, 'percentage': float}, ...]}

get_motion_breakdown(data, start_time=None, end_time=None) -> dict
  Get time spent in each motion state (walking, stationary, etc).
  Returns: {'motions': [{'motion': str, 'time_formatted': str, 'percentage': float}, ...]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRAME-LEVEL QUERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

get_frame_by_index(data, frame_index) -> dict
  Get specific frame by number.
  Returns: {'found': bool, 'frame': {...frame data...}}
  IMPORTANT: Access data via result['frame']['task_name'], result['frame']['wearer_productivity_status'], etc.

get_frame_at_time(data, time_seconds) -> dict
  Get frame closest to a timestamp.
  Returns: {'found': bool, 'frame': {...frame data...}, 'time_difference_sec': float}

filter_frames_by_status(data, status, start_time=None, end_time=None) -> dict
  Filter frames by "ACTIVE" or "IDLE".
  Returns: {'frames': [...], 'frame_count': int}

filter_frames_by_task(data, task_name, start_time=None, end_time=None) -> dict
  Filter frames by task (partial match).
  Returns: {'frames': [...], 'frame_count': int}

filter_frames_by_tool(data, tool_name, start_time=None, end_time=None) -> dict
  Filter frames by tool (partial match).
  Returns: {'frames': [...], 'frame_count': int}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

get_overall_summary(data) -> dict
  BEST FOR GENERAL QUESTIONS. Comprehensive session overview.
  Returns: {
    'total_frames': int,
    'duration_formatted': str,
    'active_percentage': float,
    'idle_percentage': float,
    'unique_tasks': [str, ...],
    'unique_tools': [str, ...],
    'unique_scenes': [str, ...],
    'task_count': int,
    'tool_count': int
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADVANCED ANALYTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

find_idle_periods(data, min_duration_seconds=2.0) -> dict
  Find CONTINUOUS idle periods (not just individual frames).
  USE FOR: "When did the worker take breaks?", "Find all idle stretches"
  Returns: {
    'idle_periods': [{'start_time_formatted', 'duration_formatted', ...}, ...],
    'total_idle_periods': int,
    'total_idle_time_formatted': str,
    'longest_idle_period': dict
  }

compare_time_periods(data, period1_start, period1_end, period2_start, period2_end) -> dict
  Compare productivity between two time periods.
  USE FOR: "Compare first half vs second half", "Was morning better than afternoon?"
  Returns: {
    'period1': {'active_percentage', 'idle_percentage', 'unique_tasks', ...},
    'period2': {...},
    'comparison': {'more_productive', 'productivity_difference', 'analysis'}
  }

detect_fatigue(data) -> dict
  Detect behavioral fatigue by comparing first vs second half.
  USE FOR: "Is the worker tired?", "Any signs of fatigue?", "Performance decline?"
  Returns: {
    'fatigue_detected': bool,
    'fatigue_level': "none" | "mild" | "moderate" | "severe",
    'productivity_drop': float,
    'idle_increase': float,
    'recommendation': str
  }

get_productivity_trend(data, num_segments=5) -> dict
  Analyze productivity trend over the session.
  USE FOR: "Is productivity improving or declining?", "What's the trend?"
  Returns: {
    'trend': "improving" | "declining" | "stable",
    'trend_strength': "strong" | "moderate" | "slight",
    'segments': [...],
    'analysis': str
  }

get_peak_productivity_period(data, window_seconds=60) -> dict
  Find the most productive period in the session.
  USE FOR: "When was peak performance?", "Best productive stretch?"
  Returns: {
    'peak_start_formatted': str,
    'peak_end_formatted': str,
    'peak_productivity': float,
    'peak_task': str,
    'comparison_to_average': float
  }

get_activity_segments(data, min_segment_duration=2.0) -> dict
  Get continuous activity segments (chunks of consistent work/idle).
  USE FOR: "How was work organized?", "Show work segments"
  Returns: {
    'segments': [{'status', 'duration_formatted', 'primary_task', 'primary_tool'}, ...],
    'total_segments': int,
    'average_segment_duration_formatted': str
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Idle time for entire video:
stats = calculate_idle_time_in_range(data, None, None)
answer = f"Idle: {stats['idle_time_formatted']} ({stats['idle_percentage']}%)"

# Find idle breaks:
result = find_idle_periods(data, min_duration_seconds=5)
answer = result

# Compare first vs second half:
summary = get_overall_summary(data)
mid = summary['duration_seconds'] / 2
result = compare_time_periods(data, 0, mid, mid, summary['duration_seconds'])
answer = result

# Check for fatigue:
result = detect_fatigue(data)
answer = result

# Get productivity trend:
result = get_productivity_trend(data)
answer = result

# Find peak performance:
result = get_peak_productivity_period(data)
answer = result

# What happened at frame 50:
result = get_frame_by_index(data, 50)
if result['found']:
    frame = result['frame']
    answer = {'task': frame['task_name'], 'status': frame['wearer_productivity_status']}
"""

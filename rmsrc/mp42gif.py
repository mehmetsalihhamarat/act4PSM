from moviepy.editor import VideoFileClip, vfx

def mp4_to_gif(input_path, output_path, speedup_factor=1.0, fps=10, scale=0.5):
    """
    Convert an MP4 video to a GIF with optional speedup, frame rate, and resolution reduction.

    Args:
        input_path (str): Path to the input MP4 file.
        output_path (str): Path to save the output GIF file.
        speedup_factor (float): Factor by which to speed up the video. Default is 1.0 (no speedup).
        fps (int): Frames per second for the GIF. Lowering this reduces file size.
        scale (float): Scaling factor for the resolution. Default 0.5 scales to 50% of the original size.
    """
    clip = VideoFileClip(input_path)
    
    # Apply speedup
    if speedup_factor != 1.0:
        clip = clip.fx(vfx.speedx, speedup_factor)
    
    # Resize to reduce resolution
    clip = clip.resize(scale)
    
    # Write GIF with reduced frame rate
    clip.write_gif(output_path, fps=fps)

# Example usage
mp4_to_gif("isaac.mp4", "isaac.gif", speedup_factor=1.5, fps=10, scale=0.8)

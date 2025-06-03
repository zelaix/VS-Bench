import os
import imageio


class Recorder:

    def __init__(self, file_dir, file_type='gif', fps=10, file_name='recording'):
        """
        Initialize the recorder
        
        Args:
            file_type (str): Type of output, either 'gif' or 'video'
            fps (int): Frames per second, default is 10
        """
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_type = file_type.lower()
        self.fps = fps
        self.frames = []

        if self.file_type not in ['gif', 'mp4']:
            raise ValueError("file_type must be either 'gif' or 'mp4'")

    # def set_recording_path(self, output_path):
    #     """
    #     Set the output path for the recording
    #     Args:
    #         output_path (str): Path to save the recording
    #     """
    #     self.output_path = output_path
    #     os.makedirs(self.output_path, exist_ok=True)

    def add_frame(self, image_path):
        """
        Add a frame to the recording
        
        Args:
            image_path (str): Path to the image file
        """
        if os.path.exists(image_path):
            self.frames.append(image_path)

    def save(self):
        """
        Save the recording file
        """
        if not self.frames:
            print("No frames to save")
            return

        self.output_file = os.path.join(self.file_dir, f'{self.file_name}.{self.file_type}')

        if self.file_type == 'gif':

            with imageio.get_writer(self.output_file, mode='I', fps=self.fps) as writer:
                for frame_path in self.frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
        else:

            with imageio.get_writer(self.output_file, fps=self.fps) as writer:
                for frame_path in self.frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

        # print(f"{self.file_type.upper()} saved to: {self.output_file}")
        self.clear()

    def clear(self):
        """
        Clear all frames
        """
        self.frames = []

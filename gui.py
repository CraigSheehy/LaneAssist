import PySimpleGUI as gui
import os.path

file_list_column = [
    [
        gui.Text("Video Folder"),
        gui.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        gui.FolderBrowse(),
    ],

    [
        gui.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
video_viewer_column = [
    [gui.Text("Choose an video from list on left:")],
    [gui.Text(size=(40, 1), key="-TOUT-")],
    [gui.Image(key="-VIDEO-")],
]

# ----- Full layout -----
layout = [
    [
        gui.Column(file_list_column),
        gui.VSeperator(),
        gui.Column(video_viewer_column),
    ]
]

window = gui.Window("Video Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == gui.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".MOV", ".mp4"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-VIDEO-"].update(filename=filename)

        except:
            pass

window.close()


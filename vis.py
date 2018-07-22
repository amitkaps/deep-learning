import numpy as np
import pandas as pd
import altair as alt

# Show an image from an nparray
def imshow(nparray, label=None, color="greys"):
    img = pd.DataFrame(nparray).reset_index().melt("index")
    img.columns = ["y" , "x", "value"]
    image = alt.Chart(img).mark_rect().encode(
        alt.X('x:N', axis=alt.Axis(title='', labelAngle=0)), 
        alt.Y('y:N', axis=alt.Axis(title='')), 
        alt.Color("value", legend=None, sort="descending", scale=alt.Scale(scheme=color)),
        tooltip = ["value"]
    ).properties(
        width = 400,
        height = 400,
        title = label
    )
    return image
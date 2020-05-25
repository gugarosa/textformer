import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_attention(input_text, translated_text, attentions, color_map='hot'):
    """Plots an attention graph between input text and translated text.

    Args:
        input_text (list): List of input tokens.
        translated_text (list): List of translated tokens.
        attentions (torch.Tensor): Tensor holding attention values for each (input, translated) token pair.
        color_map (str): A matplotlib identifier for the color mapping to be used.

    """

    # Creating a figure and its axis
    fig= plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    
    # Detaching and transferring attentions to a numpy array
    attentions = attentions.squeeze(1).detach().cpu().numpy()
    
    # Defining a color map for the attentions
    ax.matshow(attentions, cmap=color_map)
    
    # Appending `<sos>` and `<eos>` tokens to input text, as well as a rotation for making
    # a diagonal matrix
    ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in input_text] + ['<eos>'], rotation=45)

    # Appeding just an empty index to translated text for clearer visualization
    ax.set_yticklabels([''] + translated_text)

    # Fixing up the amount of ticks according to amount of tokens
    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator())

    # Showing and closing plot
    plt.show()
    plt.close()

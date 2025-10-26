import io
import json
import logging
import os
import sys
import time
from functools import lru_cache, wraps

import matplotlib.pyplot as plt
import nextcord

logger = logging.getLogger(__name__)


@lru_cache
def load_config():
    """Loads the configuration from 'config.json'."""
    if not os.path.isfile("./config.json"):
        sys.exit("'config.json' not found! Please add it and try again.")
    else:
        with open("./config.json", 'r', encoding='utf-8', ) as file:
            return json.load(file)


def embed(fig, title):
    """Converts a matplotlib figure to a nextcord Embed with an image
    attachment.
    """
    with io.BytesIO() as output:
        fig.savefig(output, format='png', bbox_inches="tight", dpi=80)
        output.seek(0)
        filename = f"{title.replace(' ', '_')[:30]}.png"
        chart = nextcord.File(output, filename=filename)
        _embed = nextcord.Embed(
            title=title[:200],
            type='image'
        )
        _embed.set_image(
            url=f"attachment://{filename}"
        )
        plt.close()
        return _embed, chart


async def load_cogs(bot, config) -> None:
    """Loads cogs from the 'cogs' directory based on the configuration."""
    cogs = config['cogs']
    for file in os.listdir("./cogs"):
        if file.endswith(".py"):
            extension = file[:-3]
            if extension in cogs:
                try:
                    bot.load_extension(f"cogs.{extension}")
                    bot.logger.info(f"Loaded extension '{extension}'")
                except Exception as e:
                    exception = f"{type(e).__name__}: {e}"
                    print(f"Failed to load extension {extension}\n{exception}")
                    bot.logger.error(
                        f"Failed to load extension {extension}\n{exception}")


def retry(
        max_tries=3,
        delay_seconds=1
):
    """Decorator to retry a function call a specified number of times"""
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(
            *args,
            **kwargs
        ):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)
        return wrapper_retry
    return decorator_retry


def error_catch():
    def decorator(func):
        @wraps(func)
        def wrapper_retry(
            *args,
            **kwargs
        ):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}", e)
        return wrapper_retry
    return decorator

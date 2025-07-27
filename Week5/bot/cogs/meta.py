import nextcord
from nextcord.ext import commands
from helpers.utils import load_config, load_cogs
config = load_config()


class Meta(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        super().__init__()
        self.bot = bot

    @nextcord.slash_command(guild_ids=[config['guild_id']], description="Delete a number of messages.")
    async def purge(self, interaction: nextcord.Interaction, amount: int = 1):
        await interaction.response.send_message(content=f'Deleting {amount} Messages...')
        purged_messages = await interaction.channel.purge(limit=amount+1)

        embed = nextcord.Embed(
            description=f"cleared **{len(purged_messages)-1}** messages!",
            color=0x9C84EF
        )
        await interaction.channel.send(embed=embed)
        # await interaction.followup.send(content="This is a slash command in a cog!")


def setup(bot):
    bot.add_cog(Meta(bot))

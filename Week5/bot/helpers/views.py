import nextcord
from nextcord import Interaction
from typing import Optional, List, Callable, Any


class EphemeralContainer(nextcord.ui.View):
    def __init__(
            self,
            items: List[nextcord.ui.View],
            *,
            timeout: Optional[float] = 180,
            auto_defer: bool = True
    ):
        super().__init__(timeout=timeout, auto_defer=auto_defer)
        for item in items:
            self.add_item(item)


class ListItem(nextcord.ui.Button["List"]):
    def __init__(
        self,
        label: str,
        modal: nextcord.ui.Modal = None,
        row: int = 0,
        callback_func: Callable[[], Any] = None
    ):
        super().__init__(
            style=nextcord.ButtonStyle.secondary,
            label="\u200b",
            row=row
        )
        self.label = label
        self.modal = modal
        self.callback_func = callback_func

    async def callback(self, interaction: Interaction):
        if self.modal is not None:
            await interaction.response.send_modal(self.modal)

        if self.callback_func is not None:
            self.callback_func()

        # await interaction.send('Closed')

        # await interaction.followup.send('Closed')

    # @nextcord.ui.button(label="Cancel", style=nextcord.ButtonStyle.grey)
    # async def dismiss(self, button: nextcord.ui.Button, interaction: nextcord.Interaction):
    #     await interaction.response.send_message("Dismissing", ephemeral=True)
    #     self.stop()

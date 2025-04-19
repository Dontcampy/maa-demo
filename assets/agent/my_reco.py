import json

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context


@AgentServer.custom_recognition("my_reco_222")
class MyRecongition(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:

        reco_detail = context.run_recognition(
            "MyCustomOCR",
            argv.image,
            pipeline_override={"MyCustomOCR": {"roi": [100, 100, 200, 300]}},
        )

        # context is a reference, will override the pipeline for whole task
        context.override_pipeline({"MyCustomOCR": {"roi": [1, 1, 114, 514]}})
        # context.run_recognition ...

        # make a new context to override the pipeline, only for itself
        new_context = context.clone()
        new_context.override_pipeline({"MyCustomOCR": {"roi": [100, 200, 300, 400]}})
        reco_detail = new_context.run_recognition("MyCustomOCR", argv.image)

        click_job = context.tasker.controller.post_click(10, 20)
        click_job.wait()

        context.override_next(argv.node_name, ["TaskA", "TaskB"])

        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 100, 100), detail="Hello World!"
        )

@AgentServer.custom_recognition("ur_rcon")
class UrReco(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        ship_rcon = {"ship_rcon":
                       {"recognition": "TemplateMatch",
                        "template": "card/feature/lv1.png",
                        "threshold":0.5,
                        "roi": [219, 88, 1014, 470]
                        }
                   }
        ship_detail = context.run_recognition(
            "ship_rcon",
            argv.image,
            pipeline_override=ship_rcon,
        )

        go_to_unit_task = {"go_to_unit_task":
            {
                "recognition": "OCR",
                "expected": "单位",
                "roi": [
                    68,
                    218,
                    53,
                    39
                ],
                "action": "Click",
                "post_wait_freezes": 2000,
            }
        }

        context.run_task(
            "go_to_unit_task",
            pipeline_override=go_to_unit_task,
        )

        unit_rcon = {"unit_rcon":
                       {"recognition": "TemplateMatch",
                        "template": "card/feature/lv1.png",
                        "threshold":0.5,
                        "roi": [219, 88, 1014, 470]
                        }
                   }

        unit_detail = context.run_recognition(
            "unit_rcon",
            context.tasker.controller.post_screencap().wait().get(),
            pipeline_override=unit_rcon,
        )

        # Trick vscode extension
        empty_rcon = {"empty_rcon":
                       {"recognition": "DirectHit"
                        }
                   }

        context.run_recognition(
            "empty_rcon",
            argv.image,
            pipeline_override=empty_rcon,
        )

        ship_count: int = 0
        if ship_detail is not None:
            ship_count = len(ship_detail.filterd_results)

        unit_count: int = 0
        if unit_detail is not None:
            unit_count = len(unit_detail.filterd_results)
        ur_count = ship_count + unit_count

        args = json.loads(argv.custom_recognition_param)
        if ur_count < int(args["ur_count"]):
            context.override_next(argv.node_name, ["open_menu"])

        # return CustomRecognition.AnalyzeResult(
        #     box=(0, 0, 0, 0), detail=f'{len(ship_detail.all_results)}, {len(ship_detail.filterd_results)}'
        # )

        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 0, 0), detail=f'ur_count:{ur_count}, unit_count:{unit_count}, ship_count:{ship_count}, {args}'
        )

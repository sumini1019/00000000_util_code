from knockknock import slack_sender

# webhook_url = "https://hooks.slack.com/services/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"
webhook_url = "https://hooks.slack.com/services/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"

print('시작')

@slack_sender(webhook_url=webhook_url, channel="#noti", user_mentions=["@정수민"])
def train_your_nicest_model(your_nicest_parameters):
    import time
    # time.sleep(10)
    return {'loss': 0.9} # Optional return value

temp = train_your_nicest_model(1)

print('끝')
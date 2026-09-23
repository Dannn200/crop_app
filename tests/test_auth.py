import unittest

from app import app


class AuthFlowTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_signup_creates_user_and_logs_in(self):
        response = self.client.post('/signup', data={
            'username': 'newuser',
            'password': 'newpass'
        }, follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn('/upload', response.headers['Location'])

        login_response = self.client.post('/', data={
            'username': 'newuser',
            'password': 'newpass'
        }, follow_redirects=False)

        self.assertEqual(login_response.status_code, 302)
        self.assertIn('/upload', login_response.headers['Location'])


if __name__ == '__main__':
    unittest.main()
